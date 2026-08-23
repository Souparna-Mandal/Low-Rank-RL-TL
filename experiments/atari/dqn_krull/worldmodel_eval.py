"""Evaluate the published DIAMOND / IRIS Atari-100k Krull checkpoints under
this repo's seeded protocol: 32 full-game episodes, reset seeds 1000 + i.

Run from experiments/atari/dqn_krull (one process per model — the two repos
have clashing module names):
    python worldmodel_eval.py diamond [--episodes N] [--video]
    python worldmodel_eval.py iris    [--episodes N] [--video]

Each model uses ITS OWN published env contract and action rule:
  * DIAMOND: KrullNoFrameskip-v4, noop<=30, max-skip 4 on native frames then
    cv2 INTER_AREA resize to 64x64 RGB, obs scaled to [-1, 1]; policy SAMPLED
    from the actor-critic logits (their test epsilon = 0).
  * IRIS: KrullNoFrameskip-v4, PIL-bilinear resize to 64x64 RGB FIRST, then
    noop<=30 and max-skip 4 over resized frames, obs in [0, 1]; action routed
    through tokenizer.encode_decode, SAMPLED at temperature 0.5 (their test
    collection settings).
Both are full-game (no life-loss termination), unclipped raw scores — the
same quantity our eval_scores.csv records.

Results -> cached/worldmodel_eval/<model>_krull.json
Videos  -> videos/<model>_score<S>.mp4 (median-scoring eval seed, --video)
"""
import argparse, json, pathlib, sys, time, types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
EXT = HERE / "cached" / "external"
OUT = HERE / "cached" / "worldmodel_eval"
OUT.mkdir(parents=True, exist_ok=True)

EPISODES, BASE_SEED, N_ACTIONS = 32, 1000, 18


def make_ale(render=False):
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    kw = dict(full_action_space=False, frameskip=1)
    if render:
        kw["render_mode"] = "rgb_array"
    try:
        return gym.make("KrullNoFrameskip-v4", **kw)
    except gym.error.Error:
        return gym.make("ALE/Krull-v5", repeat_action_probability=0.0, **kw)


# --- IRIS's exact wrapper chain (gymnasium API port of iris/src/envs/wrappers.py)
def make_iris_env(render=False):
    import gymnasium as gym
    from PIL import Image

    class PILResize(gym.ObservationWrapper):
        def __init__(self, env, size=64):
            super().__init__(env)
            self.size = (size, size)
            self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), np.uint8)

        def observation(self, obs):
            return np.array(Image.fromarray(obs).resize(self.size, Image.BILINEAR))

    class NoopReset(gym.Wrapper):
        def __init__(self, env, noop_max=30):
            super().__init__(env)
            self.noop_max = noop_max

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.env.unwrapped.np_random.integers(1, self.noop_max + 1)):
                obs, _, term, trunc, info = self.env.step(0)
                if term or trunc:
                    obs, info = self.env.reset(**kwargs)
            return obs, info

    class MaxAndSkip(gym.Wrapper):
        def __init__(self, env, skip=4):
            super().__init__(env)
            self._skip = skip
            self._buf = np.zeros((2, *env.observation_space.shape), np.uint8)

        def step(self, action):
            total, term, trunc, info = 0.0, False, False, {}
            for i in range(self._skip):
                obs, r, term, trunc, info = self.env.step(action)
                if i == self._skip - 2:
                    self._buf[0] = obs
                if i == self._skip - 1:
                    self._buf[1] = obs
                total += r
                if term or trunc:
                    break
            return self._buf.max(axis=0), total, term, trunc, info

    return MaxAndSkip(NoopReset(PILResize(make_ale(render))))


def load_diamond(device):
    import torch
    sys.path.insert(0, str(EXT / "diamond" / "src"))
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__getattr__ = lambda name: (lambda *a, **k: None)
    sys.modules.setdefault("wandb", wandb_stub)
    from models.actor_critic import ActorCritic, ActorCriticConfig

    cfg = ActorCriticConfig(lstm_dim=512, img_channels=3, img_size=64,
                            channels=[32, 32, 64, 64], down=[1, 1, 1, 1],
                            num_actions=N_ACTIONS)
    ac = ActorCritic(cfg).to(device).eval()
    sd = torch.load(EXT / "hf_diamond" / "atari_100k" / "models" / "Krull.pt",
                    map_location=device)
    ac.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                        if k.startswith("actor_critic.")})

    state = {}

    def policy(obs, reset):
        # obs: uint8 HWC RGB 64x64 -> [-1, 1] CHW (diamond TorchEnv._to_tensor)
        if reset:
            state["hx"] = torch.zeros(1, 512, device=device)
            state["cx"] = torch.zeros(1, 512, device=device)
        t = (torch.as_tensor(obs, device=device).float().div(255).mul(2).sub(1)
             .permute(2, 0, 1).unsqueeze(0))
        with torch.no_grad():
            logits, _, (state["hx"], state["cx"]) = \
                ac.predict_act_value(t, (state["hx"], state["cx"]))
            return int(torch.distributions.Categorical(logits=logits).sample())

    return policy, make_ale


def load_iris(device):
    import torch
    sys.path.insert(0, str(EXT / "iris" / "src"))
    tv_stub = types.ModuleType("torchvision")
    tv_stub.models = types.SimpleNamespace()
    sys.modules.setdefault("torchvision", tv_stub)
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)   # iris pins legacy gym; API-compatible
                                               # for the class definitions it imports
    from models.tokenizer.nets import Encoder, Decoder, EncoderDecoderConfig
    from models.tokenizer import Tokenizer
    from models.actor_critic import ActorCritic

    ed = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512,
                              ch=64, ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                              attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512, encoder=Encoder(ed),
                    decoder=Decoder(ed), with_lpips=False).to(device).eval()
    ac = ActorCritic(act_vocab_size=N_ACTIONS,
                     use_original_obs=False).to(device).eval()
    sd = torch.load(EXT / "hf_iris" / "pretrained_models" / "Krull.pt",
                    map_location=device)
    tok.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                         if k.startswith("tokenizer.")
                         and not k.startswith("tokenizer.lpips.")})
    ac.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                        if k.startswith("actor_critic.")})
    TEMPERATURE = 0.5                      # iris trainer.yaml test collection

    def policy(obs, reset):
        # obs: uint8 HWC RGB 64x64 -> [0, 1] CHW; act via tokenizer round-trip
        if reset:
            ac.reset(n=1)
        t = (torch.as_tensor(obs, device=device).float().div(255)
             .permute(2, 0, 1).unsqueeze(0))
        with torch.no_grad():
            rec = torch.clamp(
                tok.encode_decode(t, should_preprocess=True,
                                  should_postprocess=True), 0, 1)
            logits = ac(rec).logits_actions[:, -1] / TEMPERATURE
            return int(torch.distributions.Categorical(logits=logits).sample())

    return policy, make_iris_env


def wrap_diamond_env(render=False):
    from envs.atari_preprocessing import AtariPreprocessing
    return AtariPreprocessing(env=make_ale(render), noop_max=30, frame_skip=4,
                              screen_size=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["diamond", "iris"])
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--video", action="store_true",
                    help="record the median-scoring eval seed to videos/")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "diamond":
        policy, _ = load_diamond(device)
        make_env = wrap_diamond_env                  # needs diamond on sys.path
    else:
        policy, make_env = load_iris(device)

    def rollout(env, seed):
        obs, _ = env.reset(seed=seed)
        a = policy(obs, reset=True)
        total, steps, term, trunc = 0.0, 0, False, False
        while not (term or trunc):
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            steps += 1
            if not (term or trunc):
                a = policy(obs, reset=False)
        return total, steps

    env = make_env()
    scores, lengths = [], []
    for i in range(args.episodes):
        t0 = time.time()
        s, n = rollout(env, BASE_SEED + i)
        scores.append(s), lengths.append(n)
        print(f"{args.model} ep {i:2d} (seed {BASE_SEED + i}): score {s:6.0f} "
              f"len {n:5d}  {time.time() - t0:4.0f}s", flush=True)
    env.close()

    res = {"model": args.model, "game": "Krull", "episodes": args.episodes,
           "base_seed": BASE_SEED, "scores": scores, "lengths": lengths,
           "mean": float(np.mean(scores)), "median": float(np.median(scores)),
           "std": float(np.std(scores, ddof=1)),
           "min": float(np.min(scores)), "max": float(np.max(scores)),
           "protocol": ("sampled policy, eps 0" if args.model == "diamond"
                        else "sampled at temperature 0.5")}
    with open(OUT / f"{args.model}_krull.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"{args.model}: mean {res['mean']:.1f} median {res['median']:.1f} "
          f"std {res['std']:.1f}")

    if args.video:
        from gymnasium.wrappers import RecordVideo
        idx = int(np.argsort(scores)[len(scores) // 2])
        vdir = HERE / "videos"
        env = make_env(render=True)
        env = RecordVideo(env, str(vdir), name_prefix=f"tmp_{args.model}",
                          episode_trigger=lambda e: True)
        s, _ = rollout(env, BASE_SEED + idx)
        env.close()
        dst = vdir / f"{args.model}_score{int(s)}.mp4"
        (vdir / f"tmp_{args.model}-episode-0.mp4").rename(dst)
        print(f"video: eval seed {BASE_SEED + idx} -> {dst.name}")


if __name__ == "__main__":
    main()
