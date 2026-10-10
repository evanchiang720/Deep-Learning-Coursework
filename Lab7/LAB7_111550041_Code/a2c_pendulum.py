#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, out_dim)
        self.std = nn.Linear(128, out_dim)
        initialize_uniformly(self.mu, init_w=3e-3)
        initialize_uniformly(self.std, init_w=3e-3)
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = self.network(state)
        mu = self.mu(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-4#torch.exp(torch.clamp(std, -20, 2)) # 0 ~ inf 
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        initialize_uniformly(self.network[-1], init_w=3e-3)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        value = self.network(state)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.env_step = args.env_step
        
        # device: cpu / gpu
        self.device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done, terminated, truncated])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done, terminated, truncated = self.transition
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device).view(-1)

        with torch.no_grad():
            if not terminated:
                Q_t = reward + self.gamma * self.critic(next_state).view(-1)
            else:
                Q_t = reward
        mask = 1 - float(done)
        
        ############TODO#############
        V_t = self.critic(state).view(-1)
        value_loss = F.mse_loss(Q_t, V_t)
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) 
        self.critic_optimizer.step()

        advantage = (Q_t - V_t).detach()
        ############TODO#############
        _, dist = self.actor(state)
        entropy = dist.entropy().mean()
        policy_loss = -(log_prob.view(-1) * advantage.view(-1)).mean() - self.entropy_weight * entropy
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def evaluate(self, num_episodes=20):
        """Test the agent."""
        self.is_test = True
        self.actor.eval()
        eval_env = gym.make(self.env.spec.id)
        scores = []

        for i in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + 12345 + i)
            done = False
            score = 0

            while not done:
                with torch.no_grad():
                    action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                state = next_state
                score += reward

            scores.append(score)

        eval_env.close()
        self.is_test = False
        self.actor.train()
        return np.mean(scores)

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        save_interval = 10000
        eval_interval = 5000
        
        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes)):
            actor_losses, critic_losses, scores = [], [], []
            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss, entropy = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1

                if step_count % save_interval == 0:
                    checkpoint_path = f"LAB7_111550041_task1_step_{step_count}.pt"
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'step': step_count,
                        'episode': ep
                    }, checkpoint_path)
                    print(f"\nSaved model at step {step_count} to {checkpoint_path}")

                if step_count % eval_interval == 0:
                    eval_score = self.evaluate()
                    wandb.log({
                        "env_step": step_count,
                        "eval_score": eval_score
                    })

                # W&B logging
                wandb.log({
                    "env_step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "entropy": entropy
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "env_step": step_count,
                        "episode": ep,
                        "return": score
                        })  

    def test(self, video_folder: str, model_path, num_episodes=20):
        """Test the agent."""
        self.is_test = True

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            self.actor.load_state_dict(checkpoint)

        self.actor.eval()

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda x: True)

        scores = []

        for i in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + i)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            print(f"Environment steps: {self.env_step}, seed: {self.seed + i}, eval reward: {score:.2f}")
            scores.append(score)

        average_score = np.mean(scores)
        print(f"Average reward: {average_score:.2f}")
        self.env.close()
        self.env = tmp_env
        

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model-path", type=str, default="LAB7_111550041_task1_a2c_pendulum.pt")
    parser.add_argument("--video-dir", type=str, default="./video")
    parser.add_argument("--env-step", type=int, default=190000)
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    
    agent = A2CAgent(env, args)

    if args.test:
        agent.test(model_path=args.model_path, video_folder=args.video_dir)
    else:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")
        agent.train()