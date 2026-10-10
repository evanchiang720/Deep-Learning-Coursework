from diffusers import DDPMScheduler
import torch
from model import ClassConditionedUnet
from tqdm.auto import tqdm

# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def diffusion_forward(x, loss_fn, y, net):
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

    # Get the model prediction
    pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y

    # Calculate the loss
    loss = loss_fn(pred, noise) # How close is the output to the noise

    return loss

def diffusion_sample(x, y, net):
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = net(x, t, y)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    return x