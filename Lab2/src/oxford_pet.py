import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torchvision import tv_tensors

class OxfordDataset(Dataset):
    def __init__(self, img_dir, mask_dir, images, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, key):
        img_path = os.path.join(self.img_dir, f"{self.images[key]}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{self.images[key]}.png")
        image = read_image(path=img_path, mode=ImageReadMode.RGB)
        image = tv_tensors.Image(image)
        mask = read_image(path=mask_path, mode=ImageReadMode.GRAY)
        mask = tv_tensors.Mask(mask)
        image, mask = self.transform(image, mask)
        mask = (mask == 1).to(torch.float)
        mask = mask.squeeze(0)
        return image, mask
    
def loaders(train_txt, val_txt, img_dir, mask_dir, batch_size):
    def read_txt(file):
        with open(file, 'r', encoding='utf-8') as f:
            return [line.split()[0] for line in f if line.strip()]

    training_images = read_txt(train_txt)
    validating_images = read_txt(val_txt)

    training_transform = v2.Compose([
        v2.Resize(size=(512, 512), antialias=True),
        v2.ElasticTransform(alpha=50.0, sigma=5.0),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Pad(padding=92, padding_mode='reflect'), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.481, 0.449, 0.396], std=[0.228, 0.226, 0.227])
    ])
    
    validating_transform = v2.Compose([
        v2.Resize(size=(512, 512), antialias=True),
        v2.Pad(padding=92, padding_mode='reflect'), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.481, 0.449, 0.396], std=[0.228, 0.226, 0.227])
    ])

    training_set = OxfordDataset(img_dir, mask_dir, training_images, training_transform)
    validating_set = OxfordDataset(img_dir, mask_dir, validating_images, validating_transform)
    
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)
    validating_loader = DataLoader(validating_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return training_loader, validating_loader