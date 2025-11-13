import numpy as np

def extract_patches(img: np.ndarray, size: int = 256, stride: int = 256):
    """Return (patches, coords) with top-left coordinates."""
    h, w = img.shape[:2]
    patches, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = img[y:y+size, x:x+size]
            if patch.shape[0] != size or patch.shape[1] != size:
                patch = np.pad(patch,
                               ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0)),
                               mode="reflect")
            patches.append(patch)
            coords.append((y, x))
    return np.stack(patches), coords

