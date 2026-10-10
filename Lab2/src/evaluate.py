import torch
from utils import dice_score

def evaluate(model, validating_loader, device):
    total_dice_score = 0
    model.eval()
    with torch.no_grad():
        for vimages, vmasks in validating_loader:
            vimages, vmasks = vimages.to(device), vmasks.to(device)
            voutputs = model(vimages).squeeze(1)
            
            h_out, w_out = voutputs.size(-2), voutputs.size(-1)
            h_mask, w_mask = vmasks.size(-2), vmasks.size(-1)
            dy = (h_mask - h_out) // 2
            dx = (w_mask - w_out) // 2
            target = vmasks[:, dy:dy+h_out, dx:dx+w_out]
            
            preds = (voutputs > 0).float()
            total_dice_score += dice_score(preds, target)
            
    return total_dice_score / len(validating_loader)