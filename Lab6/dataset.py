import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

class IclevrDataset(Dataset):

    def __init__(self, root_dir, train_json_path, objects_json_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(train_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_names = list(self.data.keys())

        with open(objects_json_path, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = self.data[img_name]
        label_tensor = torch.zeros(len(self.classes))
        for label in labels:
            label_tensor[self.classes[label]] = 1

        return image, label_tensor