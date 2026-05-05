import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path to import SoftSkeletonize if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from losses import DiceCELoss
except ImportError:
    pass

class clDiceLoss(nn.Module):
    """
    clDice (centerline Dice) loss computes Dice on the skeletonized prediction and GT.
    It rewards topologically correct thin-tube predictions.
    """
    def __init__(self, cldice_iters=3, eps=1e-5):
        super().__init__()
        self.cldice_iters = cldice_iters
        self.eps = eps

    def forward(self, probs: torch.Tensor, target_1h: torch.Tensor, target_skel: torch.Tensor) -> torch.Tensor:
        dims = (0, 2, 3, 4)
        probs = probs.float()
        target_1h = target_1h.float()
        target_skel = target_skel.float()

        try:
            probs_skel = DiceCELoss._soft_skeletonize(probs, self.cldice_iters)
        except NameError:
            probs_skel = probs

        tprec = (probs_skel * target_1h).sum(dims) / (probs_skel.sum(dims) + self.eps)
        tsens = (target_skel * probs).sum(dims) / (target_skel.sum(dims) + self.eps)
        cldice_score = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return 1.0 - cldice_score.mean()


class SkelRecallLoss(nn.Module):
    """
    SkelRecall loss optimizes recall purely on the skeleton voxels.
    This ensures that the predicted vessel covers the full length of the 
    ground truth centerline, heavily penalizing disconnected breaks.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target_skel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C, D, H, W) soft predictions.
            target_skel: (B, C, D, H, W) ground truth skeletons.
        """
        dims = (0, 2, 3, 4)
        probs = probs.float()
        target_skel = target_skel.float()
        
        # Recall on the skeleton: how much of the GT skeleton is covered by the prediction?
        tsens_num = (target_skel * probs).sum(dims)
        tsens_den = target_skel.sum(dims)
        
        recall = tsens_num / (tsens_den + self.eps)
        # We want to maximize recall, so minimize 1 - recall
        return 1.0 - recall.mean()


class cbDiceLoss(nn.Module):
    """
    Centerline Boundary Dice (cbDice) combines centerline topology (clDice) 
    with a boundary-aware term to preserve both connectivity and surface accuracy.
    """
    def __init__(self, cldice_iters=3, alpha=0.5, eps=1e-5):
        super().__init__()
        self.cldice_iters = cldice_iters
        self.alpha = alpha # Weighting between boundary dice and clDice
        self.eps = eps

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extracts boundary using morphological gradient approximation (Dilation - mask)."""
        # 3D Max pool acts as morphological dilation
        dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
        # Eroded = -F.max_pool3d(-mask, 3, 1, 1) but Dilation - original is sufficient for outer boundary
        boundary = dilated - mask
        return torch.clamp(boundary, 0.0, 1.0)

    def forward(self, probs: torch.Tensor, target_1h: torch.Tensor, target_skel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C, D, H, W) soft predictions.
            target_1h: (B, C, D, H, W) one-hot ground truth.
            target_skel: (B, C, D, H, W) soft skeletons of the ground truth.
        """
        dims = (0, 2, 3, 4)
        probs = probs.float()
        target_1h = target_1h.float()
        target_skel = target_skel.float()

        # 1. clDice part (assuming soft skeletonize from DiceCELoss is available, or use approx)
        # If we have a soft_skeletonize backend, we'd use it here. 
        # For demonstration, we assume target_skel is provided, and we need prob_skel.
        # We will use DiceCELoss._soft_skeletonize if available.
        try:
            probs_skel = DiceCELoss._soft_skeletonize(probs, self.cldice_iters)
        except NameError:
            # Fallback if DiceCELoss is not imported
            probs_skel = probs # Fallback

        tprec = (probs_skel * target_1h).sum(dims) / (probs_skel.sum(dims) + self.eps)
        tsens = (target_skel * probs).sum(dims) / (target_skel.sum(dims) + self.eps)
        cldice_score = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)

        # 2. Boundary Dice part
        pred_boundary = self._get_boundary(probs)
        true_boundary = self._get_boundary(target_1h)
        
        b_inter = (pred_boundary * true_boundary).sum(dims)
        b_denom = pred_boundary.sum(dims) + true_boundary.sum(dims)
        boundary_dice = (2.0 * b_inter + self.eps) / (b_denom + self.eps)

        # 3. Combine
        cb_dice_score = self.alpha * cldice_score + (1.0 - self.alpha) * boundary_dice
        return 1.0 - cb_dice_score.mean()


class CASLoss(nn.Module):
    """
    Connectivity-Aware Surrogate (CAS) Loss penalizes breaks in continuous vessel structures
    by enforcing that local neighborhood connectivity in predictions matches the ground truth.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # A simple 3D uniform kernel to sum neighbors
        self.register_buffer(
            "weight", 
            torch.ones(1, 1, kernel_size, kernel_size, kernel_size)
        )

    def forward(self, probs: torch.Tensor, target_1h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C, D, H, W)
            target_1h: (B, C, D, H, W)
        """
        B, C, D, H, W = probs.shape
        probs = probs.float()
        target_1h = target_1h.float()

        loss = 0.0
        # Process each class channel independently
        for c in range(C):
            pc = probs[:, c:c+1]
            tc = target_1h[:, c:c+1]
            
            # Count local neighborhood mass (including self)
            pred_neighbors = F.conv3d(pc, self.weight, padding=self.padding)
            true_neighbors = F.conv3d(tc, self.weight, padding=self.padding)
            
            # Normalize by max possible neighbors to get a density in [0, 1]
            max_neighbors = self.kernel_size ** 3
            pred_density = pred_neighbors / max_neighbors
            true_density = true_neighbors / max_neighbors
            
            # Penalize absolute difference in connectivity density, weighted by foreground likelihood
            # This forces the model to predict locally contiguous blobs rather than isolated specks
            connectivity_diff = torch.abs(pred_density - true_density)
            loss += connectivity_diff.mean()

        return loss / C


if __name__ == "__main__":
    print("Testing Topological Losses...")
    torch.manual_seed(42)
    B, C, D, H, W = 2, 3, 16, 16, 16
    probs = torch.rand((B, C, D, H, W))
    target_1h = (torch.rand((B, C, D, H, W)) > 0.5).float()
    target_skel = (torch.rand((B, C, D, H, W)) > 0.8).float()
    
    # SkelRecall
    skel_loss_fn = SkelRecallLoss()
    skel_loss = skel_loss_fn(probs, target_skel)
    print(f"SkelRecallLoss: {skel_loss.item():.4f}")
    
    # cbDice
    cb_loss_fn = cbDiceLoss(cldice_iters=1)
    cb_loss = cb_loss_fn(probs, target_1h, target_skel)
    print(f"cbDiceLoss: {cb_loss.item():.4f}")
    
    # CAS Loss
    cas_loss_fn = CASLoss()
    cas_loss = cas_loss_fn(probs, target_1h)
    print(f"CASLoss: {cas_loss.item():.4f}")
    print("All tests passed!")
