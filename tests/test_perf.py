"""Tests for the opt-in torch.compile / autocast layer (agents/perf.py) and the
static-shape IQN target backup it enables.

The important guarantee here is that the default path is untouched: with both
knobs off, prepare_network must hand back the same object with the same bound
methods, so every config already recorded in a manifest keeps its exact eager
behaviour. The second guarantee is that the padded (static-shape) bootstrap
batch CUDA graphs require produces the same targets as the compacted one.
CPU-only; torch.compile itself is exercised only where it is cheap."""
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents.perf import prepare_network, resolve_amp_dtype   # noqa: E402
from agents.rainbow_agent import RainbowIQNNetwork           # noqa: E402


class TinyEncoder(nn.Module):
    def __init__(self, in_dim=6, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x)


def make_net(n_actions=4, seed=0):
    torch.manual_seed(seed)
    enc = TinyEncoder()
    return RainbowIQNNetwork(enc, enc.feature_dim, n_actions, n_cos=8,
                             head_hidden=16, dueling=True, noisy=False,
                             fixed_act_taus=True, n_quantiles_act=4)


# ------------------------------------------------------------- amp dtype
@pytest.mark.parametrize("spec", [None, "none", "null", "fp32", "float32"])
def test_full_precision_specs_resolve_to_none(spec):
    assert resolve_amp_dtype(spec) is None


@pytest.mark.parametrize("spec", ["bf16", "bfloat16", "BF16"])
def test_bf16_specs_resolve(spec):
    assert resolve_amp_dtype(spec) is torch.bfloat16


def test_fp16_is_rejected_with_a_scaler_explanation():
    # fp16 without a GradScaler silently underflows gradients; better to fail
    # loudly than to hand back quietly-wrong training.
    with pytest.raises(ValueError, match="GradScaler"):
        resolve_amp_dtype("fp16")


def test_unknown_dtype_rejected():
    with pytest.raises(ValueError, match="unknown amp_dtype"):
        resolve_amp_dtype("int4")


def test_torch_dtype_passes_through():
    assert resolve_amp_dtype(torch.bfloat16) is torch.bfloat16


# --------------------------------------------------------- default path
def test_both_knobs_off_is_a_true_no_op():
    net = make_net()
    original = net.quantiles
    out = prepare_network(net, compile_net=False, amp_dtype=None)
    assert out is net
    # Same bound method object -> nothing was wrapped or shadowed.
    assert net.quantiles == original
    assert "quantiles" not in net.__dict__


def test_autocast_only_wraps_without_compiling():
    net = make_net()
    prepare_network(net, compile_net=False, amp_dtype="bf16")
    assert "quantiles" in net.__dict__          # instance attr now shadows the class


def test_wrapped_network_still_produces_finite_values_of_right_shape():
    net = make_net(n_actions=4)
    prepare_network(net, compile_net=False, amp_dtype="bf16")
    x = torch.randn(5, 6)
    taus = torch.rand(5, 3)
    q = net.quantiles(x, taus)
    assert q.shape == (5, 3, 4)
    assert torch.isfinite(q.float()).all()


def test_forward_path_still_works_through_the_wrap():
    net = make_net(n_actions=4)
    prepare_network(net, compile_net=False, amp_dtype="bf16")
    q = net(torch.randn(5, 6))                  # forward -> quantiles -> mean
    assert q.shape == (5, 4)
    assert torch.isfinite(q.float()).all()


def test_parameterless_network_is_left_alone():
    class NoParams(nn.Module):
        def forward(self, x):
            return x
    net = NoParams()
    assert prepare_network(net, compile_net=True, amp_dtype="bf16") is net


# ------------------------------------ static vs compacted target backup
def test_padded_and_compacted_bootstrap_agree():
    """The compile_net path scores a zero-padded (B, ...) bootstrap batch and
    masks terminal rows; the eager path compacts to the non-final rows. Both
    index taus by the same mask, so the surviving rows must match."""
    torch.manual_seed(0)
    B, NQ, NA = 6, 3, 4
    net = make_net(n_actions=NA)
    net.eval()

    non_final = torch.tensor([True, False, True, True, False, True])
    all_next = torch.randn(B, 6)
    taus = torch.rand(B, NQ)

    with torch.no_grad():
        # compacted (default path)
        compact = all_next[non_final]
        q_c = net.quantiles(compact, taus[non_final])
        acts_c = net(compact, n_taus=NQ).argmax(1)
        gathered_c = q_c.gather(
            2, acts_c.view(-1, 1, 1).expand(-1, NQ, 1)).squeeze(2)
        theta_compact = torch.zeros(B, NQ)
        theta_compact[non_final] = gathered_c

        # padded (compile path): terminal rows zero-filled, scored, then masked
        padded = torch.zeros(B, 6)
        padded[non_final] = all_next[non_final]
        q_p = net.quantiles(padded, taus)
        acts_p = net(padded, n_taus=NQ).argmax(1)
        gathered_p = q_p.gather(
            2, acts_p.view(-1, 1, 1).expand(-1, NQ, 1)).squeeze(2)
        theta_padded = gathered_p * non_final.float().unsqueeze(1)

    assert torch.allclose(theta_compact, theta_padded, atol=1e-5), (
        "padded bootstrap batch must reproduce the compacted targets on "
        "non-final rows")


def test_all_rows_non_final_makes_the_two_paths_identical():
    """When no sampled transition is terminal the compacted batch is already
    full-width, so the static branch is selected and must be a no-op change."""
    torch.manual_seed(1)
    B, NQ, NA = 5, 3, 4
    net = make_net(n_actions=NA)
    net.eval()
    non_final = torch.ones(B, dtype=torch.bool)
    nxt, taus = torch.randn(B, 6), torch.rand(B, NQ)
    with torch.no_grad():
        a = net.quantiles(nxt, taus[non_final])
        b = net.quantiles(nxt, taus)
    assert torch.equal(a, b)


# --------------------------------------------------------------- compile
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="inductor needs a CUDA device here; the CPU backend "
                           "requires python3-dev headers (Python.h) to build")
def test_compile_preserves_values():
    """torch.compile must not change what the network computes."""
    dev = "cuda"
    net_a, net_b = make_net(seed=3).to(dev), make_net(seed=3).to(dev)
    net_b.load_state_dict(net_a.state_dict())
    prepare_network(net_b, compile_net=True, amp_dtype=None, compile_mode=None)
    x, taus = torch.randn(4, 6, device=dev), torch.rand(4, 3, device=dev)
    with torch.no_grad():
        eager, compiled = net_a.quantiles(x, taus), net_b.quantiles(x, taus)
    assert torch.allclose(eager, compiled, atol=1e-5)
