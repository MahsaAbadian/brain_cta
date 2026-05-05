import torch
import torch.nn as nn
import torch.nn.functional as F

class NexToUBlock(nn.Module):
    """
    NexToU (Neighborhood-Relation to You) architecture block.
    This block incorporates explicit neighborhood-relation modeling so the network 
    understands spatial adjacency between vessel branches. It helps correctly label 
    branching points and avoid topological errors by computing directional 
    spatial offsets (simulating graph connectivity on a voxel grid).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # We model neighborhood adjacency by taking shifted features.
        # A simple approximation of neighborhood message passing is to use 
        # a 3x3x3 depthwise convolution that learns to gather spatial context,
        # followed by a 1x1x1 pointwise convolution.
        
        self.neighborhood_gather = nn.Conv3d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1, 
            groups=in_channels, # Depthwise: independently gather spatial info per channel
            bias=False
        )
        
        self.project = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input feature map.
        Returns:
            (B, out_channels, D, H, W) adjacency-aware feature map.
        """
        # Gather information from immediate spatial neighbors
        # This acts as the neighborhood relation embedding
        neighbor_features = self.neighborhood_gather(x)
        
        # Concatenate the original center voxel features with its neighborhood context
        combined = torch.cat([x, neighbor_features], dim=1)
        
        # Project back to desired out_channels, allowing the network to fuse 
        # center knowledge with branch adjacency knowledge.
        out = self.project(combined)
        return out


class LungAirwayConnectivityModule(nn.Module):
    """
    Adapted from Lung-airway-style connectivity modules.
    Enforces long-range connectivity between predicted vessel segments by 
    applying large-kernel or dilated convolutions specifically targeted at 
    tubular structures.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Use a combination of varying dilation rates to capture long, thin
        # structures (vessels) extending far from the center voxel.
        self.branch_1 = nn.Conv3d(channels, channels // 4, kernel_size=3, padding=1, dilation=1)
        self.branch_2 = nn.Conv3d(channels, channels // 4, kernel_size=3, padding=2, dilation=2)
        self.branch_3 = nn.Conv3d(channels, channels // 4, kernel_size=3, padding=4, dilation=4)
        self.branch_4 = nn.Conv3d(channels, channels // 4, kernel_size=3, padding=8, dilation=8)
        
        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1(x)
        b2 = self.branch_2(x)
        b3 = self.branch_3(x)
        b4 = self.branch_4(x)
        
        # Concatenate multi-scale connectivity features
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        return x + self.fuse(concat) # Residual connection


if __name__ == "__main__":
    print("Testing Connectivity Architectures...")
    B, C, D, H, W = 2, 32, 16, 16, 16
    x = torch.randn(B, C, D, H, W)
    
    nextou = NexToUBlock(in_channels=C, out_channels=C)
    out_nextou = nextou(x)
    print(f"NexToUBlock output shape: {out_nextou.shape}")
    assert out_nextou.shape == x.shape
    
    airway = LungAirwayConnectivityModule(channels=C)
    out_airway = airway(x)
    print(f"LungAirwayConnectivityModule output shape: {out_airway.shape}")
    assert out_airway.shape == x.shape
    print("All tests passed!")
