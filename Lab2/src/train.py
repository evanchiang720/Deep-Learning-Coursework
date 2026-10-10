from oxford_pet import loaders
from utils import DiceLoss
from models.unet import UNet
from models.resnet34_unet import Resnet34_Unet
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import ImageFile

def train_one_epoch(training_loader, optimizer, model, device):
    scaler = GradScaler() 
    total_loss = 0
    model.train()
    pbar = tqdm(training_loader, desc="Training", unit="batch")
    
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images).squeeze(1)
            
            h_out, w_out = outputs.size(-2), outputs.size(-1)
            h_mask, w_mask = masks.size(-2), masks.size(-1)
            dy = (h_mask - h_out) // 2
            dx = (w_mask - w_out) // 2
            target = masks[:, dy:dy+h_out, dx:dx+w_out].float()

            loss = 0.5 * criterion_bce(outputs, target) + 0.5 * criterion_dice(outputs, target)
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(training_loader)

def train(model_name, model_to_train, EPOCHS):
    training_loader, validating_loader = loaders(
        train_txt='../dataset/oxford-iiit-pet/annotations/train.txt', 
        val_txt='../dataset/oxford-iiit-pet/annotations/val.txt', 
        img_dir='../dataset/oxford-iiit-pet/images', 
        mask_dir='../dataset/oxford-iiit-pet/annotations/trimaps', 
        batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_to_train().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    best_dice_score = 0
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')
        avg_loss = train_one_epoch(training_loader, optimizer, model, device)
        scheduler.step()
        
        from evaluate import evaluate
        avg_dice_score = evaluate(model, validating_loader, device)
        print(f"Training Loss: {avg_loss:.5f}, Validating Dice Score: {avg_dice_score:.5f}")
        
        if avg_dice_score > best_dice_score:
            best_dice_score = avg_dice_score
            torch.save(model.state_dict(), f'../saved_models/{model_name}_best.pth')
            print(f"--> Saved new best model with Validation Dice: {best_dice_score:.5f}")

if __name__ == "__main__":
    models = {'UNet': UNet, 'Resnet34_Unet': Resnet34_Unet}
    for model_name in models:
        train(model_name, models[model_name], EPOCHS=120)