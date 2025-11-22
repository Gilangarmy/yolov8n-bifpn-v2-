# ultralytics/nn/bifpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution - sesuai paper EfficientDet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BiFPN_Node(nn.Module):
    """Single BiFPN node dengan fast normalized fusion - sesuai paper"""
    def __init__(self, channels, num_inputs=2, epsilon=1e-4):
        super(BiFPN_Node, self).__init__()
        self.epsilon = epsilon
        self.num_inputs = num_inputs
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv = SeparableConv2d(channels, channels)
        
    def forward(self, inputs):
        # Fast normalized fusion - Equation (1) dalam paper
        w = F.relu(self.w)
        weights = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # Weighted fusion - Equation (2) dalam paper
        fused = 0
        for i in range(self.num_inputs):
            fused += weights[i] * inputs[i]
        
        # Feature refinement dengan depthwise separable conv
        return self.conv(fused)

class BiFPN_Layer(nn.Module):
    """One complete BiFPN layer - Figure 3 dalam paper"""
    def __init__(self, channels, epsilon=1e-4):
        super(BiFPN_Layer, self).__init__()
        self.channels = channels
        self.epsilon = epsilon
        
        # Top-down path nodes (P7->P6->P5->P4->P3) - sesuai paper
        self.td_p6 = BiFPN_Node(channels, 2)  # P6 + P7_up
        self.td_p5 = BiFPN_Node(channels, 2)  # P5 + P6_up  
        self.td_p4 = BiFPN_Node(channels, 2)  # P4 + P5_up
        self.td_p3 = BiFPN_Node(channels, 2)  # P3 + P4_up
        
        # Bottom-up path nodes (P3->P4->P5->P6->P7) - sesuai paper
        self.bu_p4 = BiFPN_Node(channels, 3)  # P4 + P4_td + P3_out
        self.bu_p5 = BiFPN_Node(channels, 3)  # P5 + P5_td + P4_out
        self.bu_p6 = BiFPN_Node(channels, 3)  # P6 + P6_td + P5_out
        self.bu_p7 = BiFPN_Node(channels, 2)  # P7 + P6_out
        
    def forward(self, features):
        # Unpack features: [P3, P4, P5, P6, P7] - 5 levels sesuai paper
        p3, p4, p5, p6, p7 = features
        
        # TOP-DOWN PATH - sesuai Figure 3(a)
        # P7 -> P6_td
        p7_to_p6 = F.interpolate(p7, size=p6.shape[2:], mode='nearest')
        p6_td = self.td_p6([p6, p7_to_p6])
        
        # P6_td -> P5_td  
        p6_to_p5 = F.interpolate(p6_td, size=p5.shape[2:], mode='nearest')
        p5_td = self.td_p5([p5, p6_to_p5])
        
        # P5_td -> P4_td
        p5_to_p4 = F.interpolate(p5_td, size=p4.shape[2:], mode='nearest')
        p4_td = self.td_p4([p4, p5_to_p4])
        
        # P4_td -> P3_td
        p4_to_p3 = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = self.td_p3([p3, p4_to_p3])
        
        # BOTTOM-UP PATH - sesuai Figure 3(b)
        # P3_td -> P4_out
        p3_to_p4 = F.interpolate(p3_td, size=p4.shape[2:], mode='nearest')
        p4_out = self.bu_p4([p4, p4_td, p3_to_p4])
        
        # P4_out -> P5_out
        p4_to_p5 = F.interpolate(p4_out, size=p5.shape[2:], mode='nearest')
        p5_out = self.bu_p5([p5, p5_td, p4_to_p5])
        
        # P5_out -> P6_out
        p5_to_p6 = F.interpolate(p5_out, size=p6.shape[2:], mode='nearest')
        p6_out = self.bu_p6([p6, p6_td, p5_to_p6])
        
        # P6_out -> P7_out
        p6_to_p7 = F.interpolate(p6_out, size=p7.shape[2:], mode='nearest')
        p7_out = self.bu_p7([p7, p6_to_p7])
        
        return [p3_td, p4_out, p5_out, p6_out, p7_out]

class BiFPN(nn.Module):
    """Complete BiFPN dengan 3 layers - sesuai EfficientDet-D0"""
    def __init__(self, channels, num_layers=3, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.num_layers = num_layers
        
        # ✅ SESUAI PAPER: EfficientDet-D0 menggunakan 64 channels untuk BiFPN
        bifpn_channels = 64  # Fixed sesuai paper Table 1
        
        # Projection layers untuk P3-P5 dari backbone ke BiFPN channels
        self.proj_p3 = nn.Conv2d(256, bifpn_channels, 1)   # P3: 256 -> 64
        self.proj_p4 = nn.Conv2d(512, bifpn_channels, 1)   # P4: 512 -> 64  
        self.proj_p5 = nn.Conv2d(1024, bifpn_channels, 1)  # P5: 1024 -> 64
        
        # ✅ SESUAI PAPER: Generate P6 dan P7 dari P5
        # P6 = Conv3x3 stride2(P5)
        self.p6_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, stride=2, padding=1)
        # P7 = Conv3x3 stride2(ReLU(P6))  
        self.p7_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, stride=2, padding=1)
        
        # ✅ SESUAI PAPER: 3 BiFPN layers untuk EfficientDet-D0
        self.bifpn_layers = nn.ModuleList([
            BiFPN_Layer(bifpn_channels, epsilon) for _ in range(num_layers)
        ])
        
        # Output projection kembali ke YOLO channels
        self.out_p3 = nn.Conv2d(bifpn_channels, 256, 1)  # 64 -> 256
        self.out_p4 = nn.Conv2d(bifpn_channels, 512, 1)  # 64 -> 512
        self.out_p5 = nn.Conv2d(bifpn_channels, 1024, 1) # 64 -> 1024
        
    def forward(self, inputs):
        # inputs: [P3, P4, P5] dari YOLO backbone
        p3, p4, p5 = inputs
        
        # Project ke BiFPN channels (64)
        p3_proj = self.proj_p3(p3)  # (B, 64, H/8, W/8)
        p4_proj = self.proj_p4(p4)  # (B, 64, H/16, W/16)
        p5_proj = self.proj_p5(p5)  # (B, 64, H/32, W/32)
        
        # Generate P6 dan P7 sesuai paper
        p6 = self.p6_conv(p5_proj)           # (B, 64, H/64, W/64)
        p7 = self.p7_conv(F.relu(p6))        # (B, 64, H/128, W/128)
        
        # Apply 3 BiFPN layers
        features = [p3_proj, p4_proj, p5_proj, p6, p7]
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        
        # Project kembali ke YOLO channels (hanya P3-P5 yang digunakan)
        p3_out = self.out_p3(features[0])  # P3 enhanced
        p4_out = self.out_p4(features[1])  # P4 enhanced  
        p5_out = self.out_p5(features[2])  # P5 enhanced
        
        return [p3_out, p4_out, p5_out]