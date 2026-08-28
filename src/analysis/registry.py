"""Registry mapping config method names to analysis callables, a `{name, kwargs, outputs}` entry under
`analysis.methods` / `analysis.post_methods` resolves here. It runs via the `analysis.hankel_sweep`
block dispatched inside the training loop. This is required so that config names can be converted to actual function objects
"""
from functools import partial

from analysis.low_rank.continuous_rollout import hankel_rollout_continuous
from analysis.low_rank.tabular_q_matrix import q_matrix_dqn, q_matrix_rollout, q_matrix_tabular

# name (as written in config.yaml) -> callable. The callable is invoked as
# fn(agent=agent, env=env, **kwargs) and returns one or more matrices.
ANALYSIS_METHODS = {
    "q_matrix_dqn": q_matrix_dqn,
    "q_matrix_rollout": q_matrix_rollout,
    "q_matrix_tabular": q_matrix_tabular,
    # continuous-action-only: stacked rollout Hankels of Q(s, pi(s)) and of
    # each action dimension of pi(s) — the "is the policy itself low-Hankel-
    # rank" check (raises on non-Box action spaces)
    "hankel_rollout_continuous": hankel_rollout_continuous,
}


def resolve_methods(specs):
    """Turn a list of raw config method specs into (callable, output_names) pairs.

    Each spec is `{"name": str, "kwargs": dict (optional), "outputs": list[str]}`.
    `kwargs` are bound now with functools.partial; `agent`/`env` are injected at
    call time by the caller. Accepts None/empty for configs without any methods.
    """
    return [
        (partial(ANALYSIS_METHODS[m["name"]], **m.get("kwargs", {})), m["outputs"])
        for m in (specs or [])
    ]