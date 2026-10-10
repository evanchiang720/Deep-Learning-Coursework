from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from dataset import IclevrDataset
from diffusion import diffusion_forward
from model import ClassConditionedUnet
import torch
from torch import nn
from tqdm.auto import tqdm
import random

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IclevrDataset(root_dir="./iclevr", train_json_path="./train.json", objects_json_path="./objects.json", transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Redefining the dataloader to set the batch size higher than the demo of 8
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

# How many runs through the data should we do?
n_epochs = 300

# Our network 
net = ClassConditionedUnet().to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-4) 

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        x = x.to(device)
        y = y.to(device)
        if random.random() < 0.1:
            y = torch.zeros_like(y) 

        loss = diffusion_forward(x, loss_fn, y, net)

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

    checkpoint = {
        'model': net.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'loss': avg_loss
    }
    
    torch.save(checkpoint, 'ddpm_model.pth')
    
    if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, f'ddpm_model_epoch_{epoch+1}.pth')
    print(f'Model saved at epoch {epoch}')
   