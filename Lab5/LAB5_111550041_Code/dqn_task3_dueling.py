# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import torch.nn.functional as F

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        features = self.features(x / 255.0)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(capacity * 2 - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
    
    def _propagate(self, idx, change):
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        self.data[self.write_idx] = data
        tree_idx = self.write_idx + (self.capacity - 1)
        self.update(tree_idx, p)
        self.write_idx = (self.write_idx + 1) % self.capacity
    
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = parent_idx * 2 + 1
            right_child_idx = parent_idx * 2 + 2
            if left_child_idx >= self.capacity * 2 - 1:
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 0.000001
        #self.buffer = []
        #self.priorities = np.zeros((capacity,), dtype=np.float32)
        #self.pos = 0
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.count = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        self.max_priority = max(self.max_priority, error)
        self.tree.add(error, transition)
        self.count = min(self.capacity, self.count + 1)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        data_batch, leaf_idxs, priorities = [], [], []
        segment = self.tree.total_priority / batch_size 
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            v = random.uniform(i * segment, (i + 1) * segment)
            leaf_idx, p, data = self.tree.get_leaf(v)
            data_batch.append(data)
            leaf_idxs.append(leaf_idx)
            priorities.append(p)
        
        probability = np.array(priorities) / (self.tree.total_priority + 1e-10)
        weights = (1 / (self.tree.capacity * probability)) ** self.beta
        weights /= (weights.max() + 1e-10)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return list(zip(*data_batch)), leaf_idxs, weights
    
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            epsilon = 1e-5
            p = (np.abs(error) + epsilon) ** self.alpha
            self.tree.update(idx, p)            
            self.max_priority = max(self.max_priority, p)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        
    def __len__(self):
        return self.count

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.preprocessor = AtariPreprocessor()
        self.q_net = DuelingDQN(4, self.num_actions).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net = DuelingDQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = PrioritizedReplayBuffer(args.memory_size)#deque(maxlen=args.memory_size)
        self.n_step_n = 3

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=3000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            self.n_step_buffer = deque(maxlen=self.n_step_n)

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                
                self.n_step_buffer.append((state, action, reward))
                if len(self.n_step_buffer) == self.n_step_n:
                    R = sum(self.gamma ** i * self.n_step_buffer[i][2] for i in range(self.n_step_n))
                    s0, a0, _ = self.n_step_buffer.popleft()
                    self.memory.add((s0, a0, R, next_state, done), self.memory.max_priority)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

                if self.env_count in [600000, 1000000, 1500000, 2000000, 2500000]:
                    save_path = os.path.join(self.save_dir, f"LAB5_111550041_task3_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), save_path)
                    print(f"--- Saved Task 3 Snapshot at {self.env_count} steps ---")
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            #done == True
            while len(self.n_step_buffer) > 0:
                R = sum(self.gamma ** i * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer)))
                s0, a0, _ = self.n_step_buffer.popleft()
                self.memory.add((s0, a0, R, state, True), self.memory.max_priority)

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size or self.memory.tree.total_priority == 0:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        #indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        #mini_batch = [self.memory[i] for i in indices]
        #items = list(zip(*mini_batch))
        #states, actions, rewards, next_states, dones = items
        (states, actions, rewards, next_states, dones), leaf_idxs, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q_values = rewards + (self.gamma ** self.n_step_n) * next_q_values * (1 - dones)
        errors = torch.abs(target_q_values - q_values).detach().cpu().numpy()
        self.memory.update_priorities(leaf_idxs, errors)
        #loss = (weights * (q_values - target_q_values) ** 2).mean()
        loss = (weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_task_3_dueling")
    parser.add_argument("--wandb-run-name", type=str, default="task_3_dueling")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=2500)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()