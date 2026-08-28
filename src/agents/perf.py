"""Opt-in acceleration for the Q-network stack (``torch.compile`` + autocast).

Both knobs default to OFF so every existing config stays bit-identical to the
runs already recorded in the manifests. Enable them per config under `agent`:

    agent:
      compile_net: true
      amp_dtype: bf16        # null / "fp32" keeps full precision

Measured on the GB10 (sm_121, torch 2.12+cu130) with the Atari-100k
EfficientRainbow recipe — the real ``RainbowIQNNetwork`` over
``NatureCNNEncoder`` at batch 32 x 8 taus, forward + backward + Adam:

    fp32 eager                          2.247 ms/step   1.00x
    + torch.compile (default mode)      2.054 ms/step   1.09x
    + torch.compile reduce-overhead     1.844 ms/step   1.22x
    + bf16 autocast, reduce-overhead    1.629 ms/step   1.38x

END-TO-END, HOWEVER, THIS IS A LOSS — measured on real Krull baseline runs,
same seed, compile_net+bf16 vs both off:

     8,000 env steps    off 100s    on 103s    (+3%)
    30,000 env steps    off 376s    on 391s    (+4%)

and GPU utilisation did not move (47% -> 45%). That is the whole story: the
microbenchmark speeds up the network, but the network is not what limits the
loop. A gradient step is ~6.7 ms of which the network is ~2.2 ms; the rest is
replay sampling, the uint8->device copy, DrQ augmentation, grad clipping and
the optimiser — none of which torch.compile touches. The GPU idles ~54% of
the time before and after, so making the compiled part faster just means it
waits longer.

Hence both knobs default to OFF, and turning them on is currently expected to
cost ~4%. They are kept because the picture changes on hardware where the
network *is* the constraint (an A100 saturated by several concurrent runs),
and because the plumbing is the hard part to redo.

Two further findings worth not rediscovering:

* bf16 autocast WITHOUT compile measured **0.66x** — i.e. 34% slower than
  fp32. At batch 32 the per-tensor cast overhead outweighs any tensor-core
  gain, so ``amp_dtype`` is never worth setting on its own.
* What would actually help is removing per-gradient-step Python overhead
  (batching the 64 replay draws per train() call, keeping states on device),
  not making the kernels cheaper.

Only ``quantiles`` is wrapped (falling back to ``forward`` for networks that
lack it). Acting, Double-Q selection and the target backup all funnel through
``quantiles``, so one wrap covers every hot path without nesting one compiled
region inside another.
"""

import functools

import torch

# Hot entry point, in preference order. ``RainbowIQNNetwork.forward`` and the
# Double-Q selection path both delegate to ``quantiles``, so wrapping it alone
# covers acting, selection and the TD/target forwards.
_HOT_METHODS = ("quantiles", "forward")

_AMP_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_FULL_PRECISION = (None, "none", "null", "fp32", "float32")


def resolve_amp_dtype(spec):
    """Config string -> torch dtype (or None for full precision).

    fp16 is deliberately rejected: it needs a ``GradScaler`` around the
    backward pass to avoid underflow, which is not wired into the agents.
    bf16 has the same exponent range as fp32 and needs no scaler.
    """
    if isinstance(spec, torch.dtype):
        return spec
    if spec is None or (isinstance(spec, str) and spec.lower() in _FULL_PRECISION):
        return None
    key = str(spec).lower()
    if key in _AMP_DTYPES:
        return _AMP_DTYPES[key]
    if key in ("fp16", "float16", "half"):
        raise ValueError(
            "amp_dtype='fp16' is not supported: fp16 training needs a "
            "torch.amp.GradScaler, which these agents do not wire up. Use "
            "'bf16' (same exponent range as fp32, no scaler needed).")
    raise ValueError(
        f"unknown amp_dtype {spec!r}; expected 'bf16' or null/'fp32'")


def _autocast_wrapped(fn, device_type, dtype):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=dtype):
            return fn(*args, **kwargs)
    return wrapper


def prepare_network(net, *, compile_net=False, amp_dtype=None,
                    compile_mode="reduce-overhead"):
    """Wrap ``net``'s hot entry point in autocast and/or ``torch.compile``.

    Returns ``net`` unchanged (same object, no wrapping at all) when both knobs
    are off, so the default path keeps its exact eager behaviour.

    ``compile_mode="reduce-overhead"`` enables CUDA graphs, which is where most
    of the win comes from on this workload — the GPU is starved by per-kernel
    launch latency, not short of arithmetic. CUDA graphs require a static input
    shape; see ``IQNTDMixin._target_quantiles``, which pads the bootstrap batch
    to the full batch size when compilation is on.
    """
    amp_dtype = resolve_amp_dtype(amp_dtype)
    if not compile_net and amp_dtype is None:
        return net

    try:
        device_type = next(net.parameters()).device.type
    except StopIteration:                       # parameter-less net; nothing to do
        return net

    for name in _HOT_METHODS:
        fn = getattr(net, name, None)
        if fn is None:
            continue
        if amp_dtype is not None:
            fn = _autocast_wrapped(fn, device_type, amp_dtype)
        if compile_net:
            fn = (torch.compile(fn) if compile_mode is None
                  else torch.compile(fn, mode=compile_mode))
        # Bypass nn.Module.__setattr__ so the wrapper lands in the instance
        # __dict__ and shadows the class method for every later call.
        object.__setattr__(net, name, fn)
        break                                   # only the first hot method

    return net
