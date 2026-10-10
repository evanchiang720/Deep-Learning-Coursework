import torch
import torch.nn as nn
import json
import os
from torchvision import utils
from tqdm import tqdm
import numpy as np
from diffusers import DDPMScheduler

from model import ClassConditionedUnet
from diffusion import noise_scheduler as scheduler
from evaluator import evaluation_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(img):
    return ((img + 1) / 2.0).clamp(0, 1)

def sample(net, evaluator_model, labels, save_steps, cfg_scale=1.5, classifier_scale=500.0):
    labels = labels.to(device)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    batch_size = labels.shape[0]

    x = torch.randn((batch_size, 3, 64, 64)).to(device)
    uncond_labels = torch.zeros_like(labels).to(device)

    loss_function = nn.BCELoss()
    evaluator_net = evaluator_model.resnet18

    process_images = []

    for i, t in enumerate(tqdm(scheduler.timesteps)):

        # Set requires_grad before the model forward pass
        x = x.detach().requires_grad_()
        with torch.no_grad():
            model_input = scheduler.scale_model_input(x, t)

            # predict (with grad this time)
            noise_pred_cond = net(model_input, t, labels)
            noise_pred_uncond = net(model_input, t, uncond_labels)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # Get the predicted x0:
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample
        out = evaluator_net(x0)

        # Calculate loss
        loss = loss_function(out, labels)

        # Get gradient
        cond_grad = torch.autograd.grad(loss, x)[0]

        noise_pred = noise_pred + cond_grad * classifier_scale
        x = x.detach()

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

        if i in save_steps or i == len(scheduler.timesteps)-1:
            process_images.append(denormalize(x.detach().cpu()))

    return x, process_images

def save_denoising_process(net, evaluator_model, label_list, objects_path, cfg_scale=1.5, classifier_scale=500.0):
    with open(objects_path, 'r') as f:
        objects = json.load(f)
    
    label_tensor = torch.zeros(len(objects))
    for label in label_list:
        label_tensor[objects[label]] = 1

    label_tensor = label_tensor.to(device)
    
    save_steps = np.linspace(0, len(scheduler.timesteps)-1, 11, dtype=int)

    images, process_images = sample(net, evaluator_model, label_tensor, save_steps, cfg_scale, classifier_scale)

    grid = utils.make_grid(torch.cat(process_images, dim=0), nrow=11)
    utils.save_image(grid, "process_images.png")

def run_test(net, evaluator_model, json_path, objects_path, save_dir, test_name, cfg_scale=1.5, classifier_scale=500.0):
    os.makedirs(save_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        test_data = json.load(f)
    with open(objects_path, 'r') as f:
        objects = json.load(f)

    all_labels = []
    for label_list in test_data:
        label_tensor = torch.zeros(len(objects))
        for label in label_list:
            label_tensor[objects[label]] = 1
        all_labels.append(label_tensor)
    all_labels = torch.stack(all_labels).to(device)
    
    save_steps = np.linspace(0, len(scheduler.timesteps)-1, 11, dtype=int)
    images, process_images = sample(net, evaluator_model, all_labels, save_steps, cfg_scale, classifier_scale)
    acc = evaluator_model.eval(images, all_labels)
    imgs_to_save = denormalize(images)
    for i in range(imgs_to_save.shape[0]):
        utils.save_image(imgs_to_save[i], os.path.join(save_dir, f"{i}.png"))
    
    grid = utils.make_grid(imgs_to_save, nrow=8)
    utils.save_image(grid, f"grid_{test_name}.png")
    
    return acc

if __name__ == "__main__":
    model_path = 'ddpm_model_epoch_280.pth'
    net = ClassConditionedUnet().to(device)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    evaluator = evaluation_model()

    CFG = 1.5
    CG = 800.0 

    print(f"Testing with CFG={CFG}, CG={CG}...")

    target_labels = ["red sphere", "cyan cylinder", "cyan cube"]
    save_denoising_process(net, evaluator, target_labels, "objects.json", CFG, CG)
    
    acc1 = run_test(net, evaluator, "test.json", "objects.json", "images/test", "test", CFG, CG)
    print(f"test.json Accuracy: {acc1:.4f}")

    acc2 = run_test(net, evaluator, "new_test.json", "objects.json", "images/new_test", "new_test", CFG, CG)
    print(f"new_test.json Accuracy: {acc2:.4f}")