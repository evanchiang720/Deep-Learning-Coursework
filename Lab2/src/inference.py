import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision.transforms import v2

from models.unet import UNet
from models.resnet34_unet import Resnet34_Unet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = '../dataset/oxford-iiit-pet'
IMG_DIR = os.path.join(BASE_PATH, 'images')

def rle_encode(mask):
    pixels = mask.flatten(order='F') 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_transform(size, pad_size):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(size, size), antialias=True),
        v2.Pad(padding=pad_size, padding_mode='reflect'),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.481, 0.449, 0.396], std=[0.228, 0.226, 0.227]),
    ])

def inference(model_type):
    if model_type == 'unet':
        test_txt = os.path.join(BASE_PATH, 'annotations', 'test_unet.txt')
        model = UNet(channels=3, classes=1).to(DEVICE)
        output_csv = 'unet_submission_final.csv'
        pad_size = 92
        scales_to_test = [192, 256, 320, 384, 448, 512, 576, 640, 706]
        threshold = 0.49
        model_path = '../saved_models/UNet_best.pth'
    elif model_type == 'res_unet':
        test_txt = os.path.join(BASE_PATH, 'annotations', 'test_res_unet.txt')
        model = Resnet34_Unet().to(DEVICE)
        output_csv = 'res_unet_submission_final.csv'
        pad_size = 0
        scales_to_test = [448, 512, 576]
        threshold = 0.47
        model_path = '../saved_models/Resnet34_Unet_best.pth'

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with open(test_txt, 'r') as f:
        test_list = [line.split()[0] for line in f if line.strip()]

    results = []

    with torch.no_grad():
        for img_id in tqdm(test_list):
            img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
            if not os.path.exists(img_path): 
                continue
            
            raw_image = Image.open(img_path).convert('RGB')
            w_orig, h_orig = raw_image.size
            
            accumulated_prob = torch.zeros((h_orig, w_orig), device=DEVICE)
            
            for scale in scales_to_test:
                transform = get_transform(scale, pad_size)
                input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

                out = model(input_tensor)
                prob = torch.sigmoid(out)
                
                flipped_out = model(torch.flip(input_tensor, dims=[3]))
                prob_flipped = torch.flip(torch.sigmoid(flipped_out), dims=[3])
                
                avg_scale_prob = (prob + prob_flipped) / 2.0
                
                avg_scale_prob = F.interpolate(avg_scale_prob, size=(h_orig, w_orig), mode='bilinear', align_corners=False).squeeze()
                accumulated_prob += avg_scale_prob
            
            final_prob = (accumulated_prob / len(scales_to_test)).cpu().numpy()

            results.append({
                "image_id": img_id,
                "encoded_mask": rle_encode((final_prob > threshold).astype(np.uint8))
            })
    
    pd.DataFrame(results).to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'res_unet'], required=True)
    args = parser.parse_args()
    inference(args.model)