"""Variant registry for the fable/test algorithm campaign.

Each variant is a module in this package exposing:

    build(env, device, overrides: dict) -> agent
        Agent must provide act(obs), act_and_value_only(obs), act_greedy(obs),
        update(buf), and the attribute rollout_steps — i.e. the PPOAgent
        contract from agents.ppo_agent.

    collect_rollout(agent, env, state, ep_ret)   # OPTIONAL
        Drop-in replacement for ppo_training.collect_rollout when the variant
        needs extra buffer keys (e.g. next observations per segment).

`overrides` carries hyperparameters from the run spec; unknown keys must be
consumed or ignored by the variant, never forwarded blindly to PPOAgent.
"""
import importlib

VARIANTS = ["baseline", "ssm_critic", "latent_ar", "ar_explore", "robust_hd",
            "ppg_lite", "gru_critic"]


def get_variant(name: str):
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {VARIANTS}")
    return importlib.import_module(f"agents.variants.{name}")
