from typing import Any

import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearner:
    def __init__(self, n_actions, n_bins=10, lr=0.1, gamma=0.7,
                 epsilon=1.0, eps_decay=0.999, eps_min=0.001):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma #disc factor
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        self.q_table = defaultdict(lambda: np.zeros(n_actions)) # the key is always a tuple of states
        
        self.bins = [
            np.linspace(-1, 1, n_bins),
            np.linspace(-1, 1, n_bins),
            np.linspace(-1, 1, n_bins),
            np.linspace(-1, 1, n_bins),
            np.linspace(-4* np.pi, 4*np.pi, n_bins),
            np.linspace(-9*np.pi, 9*np.pi, n_bins),
        ]
    
    def discretise(self, state):
        return tuple[Any, ...](np.digitize(state[i], self.bins[i]) for i in range(len(state)))
    
    def act(self, state, training=True): #explortation vs exploitation
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions) # we randomly choose an action 
        return np.argmax(self.q_table[self.discretise(state)]) # we choose the action with the highest Q-value
    
    def learn(self, state, action, reward, next_state, done):
        s = self.discretise(state)
        s_next = self.discretise(next_state)
        
        best_next = 0 if done else np.max(self.q_table[s_next]) # When epusode is done or goal is reched
        td_target = reward + self.gamma * best_next # reward from state + disc_factor * best Q value  Bellman Equation
        self.q_table[s][action] += self.lr * (td_target - self.q_table[s][action]) # new_value = old_value + learning_rate × (target - old_value)
    
    def decay_epsilon(self): # decaying epsilon and going to more optimal sols
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        # self.epsilon = max(self.eps_min, self.epsilon * math.exp(-1. * self.eps_decay))


def train(n_episodes=1000):
    env = gym.make("Acrobot-v1") # We have 6 states and 3 actions per state, 0 =  apply torque left, 1 = do nothing 1 = apply torque right
    agent = QLearner(n_actions=env.action_space.n, n_bins =10)
    
    rewards = []
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total += reward
        
        agent.decay_epsilon() # Ep1 = 1 ... Ep500 = 0.08
        rewards.append(total)
        
        if (ep + 1) % 100 == 0: 
            avg = np.mean(rewards[-100:])
            print(f"Episode No: {ep + 1}/{n_episodes}  Avg Reward: {avg:.1f}  Eps Value: {agent.epsilon:.3f}")
    
    env.close()
    return agent, rewards


def visualise(agent, n_episodes=3):
    env = gym.make("Acrobot-v1", render_mode="human")
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state, training=False)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total += reward
            steps += 1
        
        print(f"Episode {ep + 1}: {steps} steps, reward {total}")
    
    env.close()


def plot_progress(rewards):
    plt.figure(figsize=(12, 4))
    window = 50
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label="Reward")
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), avg, label=f"{window}-ep avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    steps = [-r for r in rewards]
    plt.plot(steps, alpha=0.3, label="Steps")
    if len(steps) >= window:
        avg = np.convolve(steps, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(steps)), avg, label=f"{window}-ep avg")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.title("Steps per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    plt.show()


# Main program

print("Training Q-learner on Acrobot...")
agent, rewards = train(n_episodes=10000)

print("\nPlotting...")
plot_progress(rewards)

print("\nVisualising trained agent...")
visualise(agent, n_episodes=3)



# Try DQN (start-with this), PPO (simple but successful)(will be good), Actor-Critic methods, TD_learning 
# Look at env where state spaces are continuous 