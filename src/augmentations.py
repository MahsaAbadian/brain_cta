import random
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter

def random_intensity_jitter(img: np.ndarray, p: float = 0.5, factor: float = 0.1) -> np.ndarray:
    """Randomly scale the intensity of the image."""
    if random.random() < p:
        scale = 1.0 + np.random.uniform(-factor, factor)
        img = img * scale
    return img

def random_gamma_augmentation(img: np.ndarray, p: float = 0.5, gamma_range: tuple[float, float] = (0.7, 1.5)) -> np.ndarray:
    """Apply random gamma correction. Expects intensities to be somewhat normalized, but will preserve sign."""
    if random.random() < p:
        gamma = np.random.uniform(*gamma_range)
        # To handle negative values which might occur after standardization,
        # we apply gamma to the absolute value and restore the sign.
        # Alternatively, for CTA resampled data, it's typically within [-1, 1] or [0, 1].
        # We'll use the safe symmetric power: sign(x) * (|x| ** gamma)
        img = np.sign(img) * (np.abs(img) ** gamma)
    return img

def random_rotation_3d(img: np.ndarray, lbl: np.ndarray, skel: np.ndarray | None = None, p: float = 0.5, max_angle: float = 15.0):
    """Randomly rotate the 3D patch along one of the three orthogonal planes."""
    if random.random() < p:
        angle = np.random.uniform(-max_angle, max_angle)
        # Randomly choose a plane to rotate (xy, xz, yz)
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        
        # Order=3 (cubic spline) for image, Order=0 (nearest-neighbor) for label and skeleton
        img = rotate(img, angle, axes=axes, reshape=False, order=3, mode='reflect', prefilter=True)
        lbl = rotate(lbl, angle, axes=axes, reshape=False, order=0, mode='reflect')
        
        if skel is not None:
            # skel is expected to be (C, D, H, W) where rotation axes need to be offset by 1
            skel_axes = (axes[0] + 1, axes[1] + 1)
            skel = rotate(skel, angle, axes=skel_axes, reshape=False, order=0, mode='reflect')
            
    return img, lbl, skel

def random_elastic_deformation_3d(img: np.ndarray, lbl: np.ndarray, skel: np.ndarray | None = None, p: float = 0.5, alpha: float = 10.0, sigma: float = 3.0):
    """Apply random elastic deformation in 3D using dense displacement fields."""
    if random.random() < p:
        shape = img.shape
        # Create random displacement fields
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # Create meshgrid of coordinates
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        # Add displacements
        indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Map coordinates
        img = map_coordinates(img, indices, order=3, mode='reflect').reshape(shape)
        lbl = map_coordinates(lbl, indices, order=0, mode='reflect').reshape(shape)

        if skel is not None:
            num_classes = skel.shape[0]
            new_skel = np.zeros_like(skel)
            for c in range(num_classes):
                new_skel[c] = map_coordinates(skel[c], indices, order=0, mode='reflect').reshape(shape)
            skel = new_skel

    return img, lbl, skel
