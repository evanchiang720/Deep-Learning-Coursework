import torch
import torch.nn as nn

def dice_score(pred, ground):
    smooth = 1e-6
    intersection = (pred * ground).sum()
    union = pred.sum() + ground.sum()
    return (2. * intersection + smooth) / (union + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice