from .base_agent import BaseAgent, EpsilonGreedyExplorer
from .perf import prepare_network
from utils.device import resolve_device

import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import Callable

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    """
    Why do we need this? 
        1) SGD relies of uncorreleated data, while policy trajectories or agent experiences
        (State, Action, Reward, Next State) is highly correlated for S(t-1) -> S(t), this usually breaks the 
        i.i.d data requirements within a batch required for SGD. Basically we'll get biased gradient estimates
        
        2) Stabilizes training distribution
        On-policy data shifts rapidly as the policy changes, causing the network to chase a moving 
        target. The buffer mixes recent transitions with older ones, 
        smoothing the training distribution and reducing catastrophic 
        forgetting of earlier-learned behaviors.
        
        3) Sample efficiency / data reuse
        Transitions (s, a, r, s') are expensive to collect. The buffer lets us reuse
        each transition across many gradient updates instead of discarding after one use.

        4) Enables off-policy learning
        DQN learns the greedy policy while acting ε-greedily. Stored transitions come
        from older policies, but Q-learning's bootstrap target r + gamma* max_a Q(s', a)
        is valid regardless of the behavior policy that generated the sample.

        5) Smooths out rare events
        Important but infrequent transitions (large rewards, goal states) stay in the
        buffer and keep influencing updates, instead of being seen once and forgotten.
    """
    def __init__(self, capacity: int):
        self._memory: deque[Transition] = deque([], maxlen=capacity) # Python deque automatically handles adding new elements once the buffer is full by discarding old ones

    def push(self, state, action, next_state, reward) -> None:
        self._memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)
    
    
class DqnLoss(nn.Module):
    """Loss function for DQN that takes in the current Q value, Reward and the current next predicted Q value to
    calculate the Target Value and then report the loss with regards to it based on the base_loss.

    """
    def __init__(self, discount_factor, base_loss = nn.HuberLoss):
        super().__init__()
        self.gamma = discount_factor
        self.base_loss =  base_loss()
    
    def forward(self, cur_Q: torch.tensor, reward: torch.Tensor, next_best_Q: torch.tensor):
        # Note all of these cur_Q, rewards ad next_Q must have a batch dimension
        target_Q = reward + self.gamma * next_best_Q 
        loss =  self.base_loss(target_Q, cur_Q)
        return loss

class QAgent(BaseAgent, EpsilonGreedyExplorer):
    """This is an agent that learns with Deep Q Learning

    Args:
        BaseAgent (_type_): _description_
        EpsilonGreedyExplorer (_type_): _description_
    """
    def __init__(
        self, replay_buffer_capacity: int, 
        q_network: Callable[..., nn.Module], batch_size: int, nn_learning_rate: float, nn_extra_kwargs: dict, 
        env: gym.Env,
        eps_start: float , eps_min: float, decay_rate: float,
        discount_factor: float, base_loss = nn.HuberLoss,
        device="auto", TD_LR = 0.1, buffer_util=1, gd_steps_ceil = 100,
        grad_clip_norm = 10.0, double=False,
        weight_decay=0.01, adam_eps=1e-8, amsgrad=True,
        compile_net=False, amp_dtype=None, compile_mode="reduce-overhead"):
        
        """Constructor for the DQN agent which considers exploration strategy, Neural Network Parameters for estimating the Q
        function along with the replay buffer for sampling from prior experiences. 

        Args:
            replay_buffer_capacity (int): _description_
            q_network (Callable[..., nn.Module]): _description_
            batch_size (int): _description_
            nn_learning_rate (float): _description_
            nn_extra_kwargs (dict): _description_
            env (gym.Env): _description_
            eps_start (float): _description_
            eps_min (float): _description_
            decay_rate (float): _description_
            discount_factor (float): _description_
            base_loss (_type_, optional): _description_. Defaults to nn.HuberLoss.
            device (str, optional): "auto", "cuda", "mps" or "cpu". "auto" and any
                unavailable choice resolve in priority order cuda > mps > cpu.
                Defaults to "auto".
            TD_LR (float, optional): _description_. Defaults to 0.1.
            buffer_util (int, optional): _description_. Defaults to 1.
            gd_steps_ceil (int, optional): capping the maximum number of gradient steps. Defaults to 100
            grad_clip_norm (float, optional): max global L2 norm for gradient clipping per step. Defaults to 10.0
            gd_steps_ceil (int, optional): capping the maximum number of gradient steps. Defaults to 100
            grad_clip_norm (float, optional): max global L2 norm for gradient clipping per step. Defaults to 10.0
            Double (Bool, optional): If enabled it uses Double DQN for Q value target estimation
            weight_decay / adam_eps / amsgrad: AdamW hyperparameters. Defaults
                keep the historical behaviour (AdamW's implicit wd=0.01,
                eps=1e-8, amsgrad on); data-efficient Atari configs set
                wd=0, eps=1.5e-4, amsgrad off (the DrQ/Dopamine values).
        """
        
        BaseAgent.__init__(self, env)
        EpsilonGreedyExplorer.__init__(self, eps_start, eps_min, decay_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        device = resolve_device(device)

        # Preparing the Neural networks for DQN
        self.policy_net = q_network(**nn_extra_kwargs).to(device)
        self.target_net = q_network(**nn_extra_kwargs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Opt-in torch.compile / autocast (both default off -> untouched eager
        # path). Applied after the state_dict copy so both nets wrap identical
        # weights. compile_net also switches the IQN target backup to a static
        # bootstrap batch, which CUDA graphs require — see perf.prepare_network.
        self.compile_net = bool(compile_net)
        self.amp_dtype = amp_dtype
        for _net in (self.policy_net, self.target_net):
            prepare_network(_net, compile_net=self.compile_net,
                            amp_dtype=amp_dtype, compile_mode=compile_mode)
        
        self.optimiser = optim.AdamW(self.policy_net.parameters(), lr=nn_learning_rate,
                                     amsgrad=amsgrad, weight_decay=weight_decay,
                                     eps=adam_eps)
        self.device = device
        self.loss = DqnLoss(discount_factor, base_loss)
        self.batch_size =  batch_size
        self.TD_LR = TD_LR
        self.buffer_util = buffer_util
        self.double = double
        self.gd_steps_ceil = gd_steps_ceil
        self.grad_clip_norm = grad_clip_norm
        
    def act_greedy(self, state: torch.tensor):
        """_summary_

        Args:
            state (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        # assuming state is in the correct device 
        return self.policy_net(state).argmax(dim=1).item()
        
    def pi(self, state: np.ndarray):
        """_summary_

        Args:
            state (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        action = None
        if self.is_random_step():
            action = self.agent_env.action_space.sample()
        else:
            # For now I am just assuming a direct integer action but change this later 
            state = torch.tensor(state, dtype= torch.float32, device=self.device).unsqueeze(0)
            action = self.act_greedy(state)
        return action
    
    def update_buffer_atari(self, state, action, reward, next_state, terminated, truncated=False):
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            terminated (_type_): _description_
        """
        state = torch.tensor(state, dtype= torch.uint8).unsqueeze(0) # adding batches
        action = torch.tensor([action], dtype= torch.long, device=self.device) # Don't make it floating point
        reward = torch.tensor([reward], dtype= torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype= torch.uint8).unsqueeze(0) if not terminated else None # adding batches
        
        self.replay_buffer.push(state,action,next_state,reward) 
    
    def update_buffer(self, state, action, reward, next_state, terminated, truncated=False):
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            terminated (_type_): _description_
        """
        state = torch.tensor(state, dtype= torch.float32, device=self.device).unsqueeze(0) # adding batches
        action = torch.tensor([action], dtype= torch.long, device=self.device) # Don't make it floating point
        reward = torch.tensor([reward], dtype= torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype= torch.float32, device=self.device).unsqueeze(0) if not terminated else None # adding batches
        
        self.replay_buffer.push(state,action,next_state,reward) 
    
    def train(self,):
        """_summary_
        """
        gd_steps = min(len(self.replay_buffer) // (self.buffer_util * self.batch_size), self.gd_steps_ceil)
        for _ in range(gd_steps):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions)) # This is a nice hack that does a transpose giving us: 
            # batch = Transition(
            #   state=(s1, s2, s3),
            #   action=(a1, a2, a3),
            #   next_state=(ns1, ns2, ns3),
            #   reward=(r1, r2, r3)
            # )
            # retriving the tensors from the transpose
            states = torch.cat(batch.state).to(self.device)
            rewards = torch.cat(batch.reward).to(self.device)
            actions = torch.cat(batch.action).to(self.device)
            non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device)
            
            Q_s_a = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # assuming the actions are integer and valid indices. We also have a batch dimension before
            Q_s_1_a = torch.zeros(self.batch_size, device=self.device)
            
            if non_final_mask.any():
                next_state = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
                with torch.no_grad():
                    if self.double:
                        next_actions = self.policy_net(next_state).argmax(dim=1, keepdim=True) # Policy net chooses the next action (max)
                        Q_s_1_a[non_final_mask] = self.target_net(next_state).gather(1, next_actions).squeeze(1) # Target net calculates the Q value for the update
                    else:
                        Q_s_1_a[non_final_mask] = self.target_net(next_state).max(dim=1).values
            
            self.optimiser.zero_grad()
            loss = self.loss(Q_s_a, rewards, Q_s_1_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
            self.optimiser.step()
    
    def update_target_network(self,):
        """_summary_
        """
        target_net_state = self.target_net.state_dict()
        policy_net_state = self.policy_net.state_dict()
        for key in policy_net_state:
            target_net_state[key] += self.TD_LR*(policy_net_state[key] - target_net_state[key])
        self.target_net.load_state_dict(target_net_state)
        
    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.epsilon = ckpt["epsilon"]
