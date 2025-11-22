"""
BiFPN asli (simplified, faithful implementation) untuk integrasi ke YOLOv8 (Ultralytics).
File ini mendefinisikan:
- SeparableConvBlock: depthwise separable conv + BN + SiLU
- WeightedAdd: fast normalized weighted fusion (ReLU->normalize)
- BiFPNBlock: satu iterasi top-down + bottom-up
- BiFPN: stack beberapa BiFPNBlock (repeats)

Catatan integrasi:
- Masukkan file ini ke: ultralytics/nn/bifpn.py
- Import di tasks.py: from ultralytics.nn.bifpn import BiFPN
- Pastikan fitur masuk (P3,P4,P5) memiliki jumlah channel yang sama (biasanya via 1x1 conv sebelumnya) — saya menambahkan ops untuk menyesuaikan channel jika perlu.
- Di YAML, ganti node-concat/concat2/concat3 yang Anda buat sebelumnya dengan 1 modul BiFPN yang mengambil list fitur input. Contoh penggunaan di parse_model: treat BiFPN as a module that consumes multiple feature maps and outputs same-numbered maps.

Implementasi ini berfokus ke kejelasan dan kompatibilitas dengan pipeline PyTorch/Ultralytics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    """Depthwise separable conv -> BN -> SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class WeightedAdd(nn.Module):
    """Fast normalized fusion used in BiFPN.
    Uses ReLU on weights then normalizes by sum + eps.
    Accepts a list of tensors and returns weighted sum.
    """
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        # initialize with equal importance
        w = torch.ones(num_inputs, dtype=torch.float32)
        self.w = nn.Parameter(w)

    def forward(self, inputs):
        # inputs: list of tensors
        w = F.relu(self.w)
        weight = w / (torch.sum(w) + self.eps)
        out = 0
        for i, t in enumerate(inputs):
            out = out + weight[i] * t
        return out


class BiFPNBlock(nn.Module):
    """
    Single BiFPN block (one top-down pass, one bottom-up pass).
    Expects a list of feature maps ordered from smallest stride (P3) -> larger strides (P4, P5)
    but we'll write to accept inputs as [P3, P4, P5] (P3 highest resolution).
    All features are expected to have the same number of channels. If not, BiFPN will
    adapt by a 1x1 conv to match channels.
    """

    def __init__(self, channels, conv_type=SeparableConvBlock):
        super().__init__()
        C = channels
        # fusion weights
        self.w1 = WeightedAdd(2)  # for top-down merges with 2 inputs
        self.w2 = WeightedAdd(3)  # for bottom-up merges with 3 inputs

        # convs after fusion
        self.p3_td_conv = conv_type(C, C)
        self.p4_td_conv = conv_type(C, C)
        self.p5_td_conv = conv_type(C, C)

        self.p3_bu_conv = conv_type(C, C)
        self.p4_bu_conv = conv_type(C, C)
        self.p5_bu_conv = conv_type(C, C)

        # if needed, adapt channels of inputs
        self.adapt_convs = None

    def adapt_input(self, inputs, channels):
        """Return inputs all adapted to `channels` using 1x1 conv if necessary."""
        adapted = []
        if self.adapt_convs is None:
            self.adapt_convs = nn.ModuleList()
            for t in inputs:
                c = t.shape[1]
                if c != channels:
                    self.adapt_convs.append(nn.Conv2d(c, channels, 1, 1, 0, bias=False))
                else:
                    self.adapt_convs.append(nn.Identity())
        for conv, t in zip(self.adapt_convs, inputs):
            adapted.append(conv(t))
        return adapted

    def forward(self, inputs):
        # inputs: [P3, P4, P5] where P3 has highest spatial resolution
        assert len(inputs) == 3, "BiFPNBlock currently supports exactly 3 levels (P3,P4,P5)"
        # adapt channels if needed
        C = inputs[0].shape[1]
        inputs = self.adapt_input(inputs, C)
        p3, p4, p5 = inputs

        # top-down pathway
        p5_up = F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='nearest')
        p4_td = self.w1([p4, p5_up])
        p4_td = self.p4_td_conv(p4_td)

        p4_up = F.interpolate(p4_td, size=(p3.shape[2], p3.shape[3]), mode='nearest')
        p3_td = self.w1([p3, p4_up])
        p3_td = self.p3_td_conv(p3_td)

        # bottom-up pathway
        p3_down = F.max_pool2d(p3_td, kernel_size=2)
        # combine p3_down, p4, p4_td (three inputs) -> p4_bu
        p4_bu = self.w2([p4, p4_td, p3_down])
        p4_bu = self.p4_bu_conv(p4_bu)

        p4_down = F.max_pool2d(p4_bu, kernel_size=2)
        p5_bu = self.w1([p5, p4_down])
        p5_bu = self.p5_bu_conv(p5_bu)

        return [p3_td, p4_bu, p5_bu]


class BiFPN(nn.Module):
    """
    Stack `num_layers` of BiFPNBlock. Input: list of feature maps [P3,P4,P5].
    All outputs keep same channel count as inputs (after optional adapt conv).

    Args:
        channels: number of channels to use internally (if inputs differ, they're adapted)
        num_layers: number of stacked BiFPN blocks (typical: 2 or 3)
    """

    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([BiFPNBlock(channels) for _ in range(num_layers)])

    def forward(self, inputs):
        """inputs: list/tuple of 3 tensors [P3,P4,P5]"""
        feats = inputs
        for b in self.blocks:
            feats = b(feats)
        return feats


# === Minimal test snippet (only runs if file executed directly) ===
if __name__ == '__main__':
    # quick smoke test
    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    bifpn = BiFPN(256, num_layers=2)
    out = bifpn([p3, p4, p5])
    for i, o in enumerate(out):
        print(f'out[{i}]', o.shape)

