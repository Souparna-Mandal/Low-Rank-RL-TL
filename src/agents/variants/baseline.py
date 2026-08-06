"""Vanilla PPO-clip baseline (the control arm)."""
from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm"}


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return PPOAgent(env=env, device=device, **kwargs)
