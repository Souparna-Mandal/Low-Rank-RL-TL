from agents import q_agent
from analysis.low_rank import rank, tabular_q_matrix, hankel_policy
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def dqn_training_loop(agent: q_agent.QAgent, env: gym.Env,
                      no_episodes: int, target_network_update_steps: int ,
                      np_seed: int = 52, no_eps_to_avg: int = 10,
                      analysis_config: dict = {},
                      DEBUG=False):
    state, info = env.reset(seed = np_seed)
    j = 0
    episode_rewards_training = []
    for episode in tqdm(range(no_episodes)):
        # Do a policy rollout and explore with the current agent
        epsiode_total_reward = 0
        terminated = truncated = False
        # This is one rollout from the current policy pi = DQN(theeta) 
        # which is updated once every episode
        while not (terminated or truncated):
            action = agent.pi(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_buffer(state, action, reward, next_state, terminated)
            state = next_state
            epsiode_total_reward += reward
            
            # This is used to bring the Network used to Calculate Q-Targets uptodate with the policy network
            j+= 1
            if j >= target_network_update_steps:
                j= 0
                agent.update_target_network()
            
        # train the agent 
        agent.train() # train once an episode finished
        # reset the environment for next episode
        state, _ = env.reset()
        episode_rewards_training.append(epsiode_total_reward)
        if episode % no_eps_to_avg == 0:
            ep_avg = sum(episode_rewards_training[-no_eps_to_avg:]) / float(no_eps_to_avg) 
            print(f"episode {episode} avg_rewarg: {ep_avg}")
        
        if DEBUG:
            print(f"episode {episode} is complete with reward{epsiode_total_reward}")
            print(f"DEBUG INFO: replay_buffer_length{len(agent.replay_buffer)}")
            
        # Analysis prints during training
        if episode % analysis_config["ep_freq"] == 0:
            for method,name in zip(analysis_config["methods"], ['Hankel Value Function','Q-function']):
                print(name)
                matrix = method(agent=agent, env=env)
                rank.plot_matrix_spectra(matrix)
            
    return episode_rewards_training