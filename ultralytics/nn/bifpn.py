# ultralytics/nn/bifpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
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
    """Single BiFPN node with fast normalized fusion"""
    def __init__(self, channels, num_inputs=2, epsilon=1e-4):
        super(BiFPN_Node, self).__init__()
        self.epsilon = epsilon
        self.num_inputs = num_inputs
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv = SeparableConv2d(channels, channels)
        
    def forward(self, inputs):
        # Fast normalized fusion
        w = F.relu(self.w)
        weights = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # Weighted fusion
        fused = 0
        for i in range(self.num_inputs):
            fused += weights[i] * inputs[i]
        
        # Feature refinement
        return self.conv(fused)

class BiFPN_Layer(nn.Module):
    """One complete BiFPN layer with bidirectional connections"""
    def __init__(self, channels, epsilon=1e-4):
        super(BiFPN_Layer, self).__init__()
        self.channels = channels
        self.epsilon = epsilon
        
        # Top-down path nodes (P6->P5->P4->P3)
        self.td_p5 = BiFPN_Node(channels, 2)
        self.td_p4 = BiFPN_Node(channels, 2) 
        self.td_p3 = BiFPN_Node(channels, 2)
        
        # Bottom-up path nodes (P3->P4->P5->P6)
        self.bu_p4 = BiFPN_Node(channels, 3)
        self.bu_p5 = BiFPN_Node(channels, 3)
        self.bu_p6 = BiFPN_Node(channels, 2)
        
    def forward(self, features):
        # Unpack features: [p3, p4, p5, p6, p7] format untuk BiFPN asli
        p3, p4, p5, p6, p7 = features
        
        # TOP-DOWN PATH
        # P7 -> P6_td
        p6_td = self.td_p5([p6, F.interpolate(p7, size=p6.shape[2:], mode='nearest')])
        
        # P6_td -> P5_td  
        p5_td = self.td_p5([p5, F.interpolate(p6_td, size=p5.shape[2:], mode='nearest')])
        
        # P5_td -> P4_td
        p4_td = self.td_p4([p4, F.interpolate(p5_td, size=p4.shape[2:], mode='nearest')])
        
        # P4_td -> P3_td
        p3_td = self.td_p3([p3, F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')])
        
        # BOTTOM-UP PATH  
        # P3_td -> P4_out
        p4_out = self.bu_p4([
            p4, 
            p4_td,
            F.interpolate(p3_td, size=p4.shape[2:], mode='nearest')
        ])
        
        # P4_out -> P5_out
        p5_out = self.bu_p5([
            p5,
            p5_td, 
            F.interpolate(p4_out, size=p5.shape[2:], mode='nearest')
        ])
        
        # P5_out -> P6_out
        p6_out = self.bu_p6([
            p6,
            F.interpolate(p5_out, size=p6.shape[2:], mode='nearest')
        ])
        
        # P6_out -> P7_out
        p7_out = p7  # P7 tetap sama dalam implementasi sederhana
        
        return [p3_td, p4_out, p5_out, p6_out, p7_out]

class BiFPN(nn.Module):
    """Complete BiFPN dengan multiple layers"""
    def __init__(self, channels_list, num_layers=2, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.num_layers = num_layers
        
        # Projection layers untuk menyesuaikan channel dimensions
        self.proj_p3 = nn.Conv2d(channels_list[0], channels_list[3], 1)
        self.proj_p4 = nn.Conv2d(channels_list[1], channels_list[3], 1) 
        self.proj_p5 = nn.Conv2d(channels_list[2], channels_list[3], 1)
        
        # Create P6 and P7 dari P5
        self.p6_conv = nn.Conv2d(channels_list[2], channels_list[3], 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(channels_list[3], channels_list[3], 3, stride=2, padding=1)
        
        # Multiple BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPN_Layer(channels_list[3], epsilon) for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_p3 = nn.Conv2d(channels_list[3], channels_list[0], 1)
        self.out_p4 = nn.Conv2d(channels_list[3], channels_list[1], 1)
        self.out_p5 = nn.Conv2d(channels_list[3], channels_list[2], 1)
        
    def forward(self, inputs):
        # inputs: [p3, p4, p5] dari backbone
        p3, p4, p5 = inputs
        
        # Project features ke dimension yang sama
        p3 = self.proj_p3(p3)
        p4 = self.proj_p4(p4) 
        p5 = self.proj_p5(p5)
        
        # Create P6 and P7
        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(F.silu(p6))
        
        # Apply BiFPN layers
        features = [p3, p4, p5, p6, p7]
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        
        # Project kembali ke original dimensions
        p3_out = self.out_p3(features[0])
        p4_out = self.out_p4(features[1]) 
        p5_out = self.out_p5(features[2])
        
        return [p3_out, p4_out, p5_out]