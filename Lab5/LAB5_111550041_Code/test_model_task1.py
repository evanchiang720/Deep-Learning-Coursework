import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
import argparse

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def evaluate_cartpole(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    num_actions = env.action_space.n

    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)
        print(f"Seed: {args.seed + ep}, eval reward: {ep_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward: {sum(total_rewards) / len(total_rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate_cartpole(args)