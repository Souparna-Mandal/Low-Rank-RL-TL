import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from agents.q_agent import QAgent


def record_greedy_episode(agent: QAgent, eval_env: gym.Env, video_dir: str,
                          episode: int, seed: int = 52, max_steps: int = 10000) -> str:
    """Roll out the agent's greedy policy for one episode and save it as an mp4.

    `eval_env` must be created with render_mode="rgb_array". Returns the video prefix
    passed to RecordVideo (the actual file is `<video_dir>/ep<episode>-episode-0.mp4`).
    """
    prefix = f"ep{episode}"
    env = RecordVideo(eval_env, video_dir, name_prefix=prefix,
                      episode_trigger=lambda e: True)
    state, _ = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            action = agent.act_greedy(state_t)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
    env.close()
    return prefix