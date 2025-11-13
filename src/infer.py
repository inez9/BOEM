import torch
import numpy as np
from torchvision.transforms import functional as TF

#1. infer.py converts image patches into tensors
#2. → runs them through model → 
#3. returns both hard predictions and confidence levels for visualization.

def _to_tensor_batch(patches: np.ndarray) -> torch.Tensor:
    #converts all image patches into a single pyTorch tensor batch
    tensors = [TF.to_tensor(p) for p in patches]
    return torch.stack(tensors)

def predict_patches(model, device, patches, batch_size=32):
    #Run all the patches through the model and get predictions 
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = _to_tensor_batch(patches[i:i+batch_size]).to(device)
            logits = model(batch)
            logits_list.append(logits.cpu())
    logits = torch.cat(logits_list)
    #torch.cat = literally just appends stuff to a list 
    probs = torch.softmax(logits, dim=1).numpy() #normalization
    labels = probs.argmax(axis=1)
    return labels, probs

