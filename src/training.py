from agents import q_agent
from analysis.low_rank import rank, tabular_q_matrix, hankel_policy
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def dqn_training_loop(agent: q_agent.QAgent, env: gym.Env,
                      no_episodes: int, target_network_update_steps: int ,
                      train_frequency_steps: int,
                      np_seed: int = 52, no_eps_to_avg: int = 10,
                      analysis_config: dict = {},
                      DEBUG=False):
    state, info = env.reset(seed = np_seed)
    j,k = 0,0
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
            
            # This is used to bring the Network used to Calculate Q-Targets up to date with the policy network
            j+= 1 
            k+=1
            if j >= target_network_update_steps:
                j = 0
                agent.update_target_network()
            if k >= train_frequency_steps:
                k = 0
                # train the agent 
                agent.train() # train every train_frequency_steps steps

        # reset the environment for next episode
        state, _ = env.reset()
        episode_rewards_training.append(epsiode_total_reward)
        if episode % no_eps_to_avg == 0:
            ep_avg = sum(episode_rewards_training[-no_eps_to_avg:]) / float(no_eps_to_avg) 
            print(f"episode {episode} avg_rewarg: {ep_avg}")
        
        if DEBUG:
            print("**************************** DEBUG INFO START **********************************")
            print(f"Episode {episode} is complete with reward {epsiode_total_reward}")
            print(f"Replay_buffer_length {len(agent.replay_buffer)}")
            print(f"Epsilon for epsilon-greedy is {agent.epsilon}")
            print("**************************** DEBUG INFO END **********************************")
            
        # Analysis prints during training
        if episode % analysis_config["ep_freq"] == 0:
            for method,name in zip(analysis_config["methods"], ['Hankel Value Function','Q-function']):
                print(f"****************************{name}****************************")
                matrix = method(agent=agent, env=env)
                r, sr, shape, irs, rc, nzc, nzr = rank.row_rank_property_check(matrix, name)
                # returns effective_rank, stable_rank, shape, irs (normalised top-r leverage per row),
                # rc = coherence of the rank-r row space ((m/rank)*max leverage, in [1, m/rank]),
                # nzc and nzr are the number of non zero columns and rows in the original matrix.
                print(f"eff_rank: {r}, stable_rank: {sr:.2f}, shape: {shape}, non-zero rows :{nzr}, non-zero cols:{nzc}")
                print(f"top-r leverage spread: min={irs.min():.4g} max={irs.max():.4g} (uniform would be {1.0/shape[0]:.4g})")
                print(f"row-space coherence score: {rc:.4g}")
            # The Hankel analysis above rolls out `env` to termination, leaving it in a stale/terminated
            # state. Reset before the next training episode so we don't resume from a hijacked env.
            state, _ = env.reset()

    return episode_rewards_training