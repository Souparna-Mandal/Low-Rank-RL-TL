"""Atari-100k launcher for the FHR-DQN comparison — full 26-game suite (+ Enduro).

Protocol (see src/analysis/atari100k.py for references and caveats):
  * 100,000 agent-environment interactions per run (400k frames, frameskip 4),
    enforced by training.max_env_steps — training stops mid-episode at the cap.
  * Deterministic ALE (repeat_action_probability 0 — the benchmark predates
    the v5 sticky default) with the per-game minimal action set.
  * Sign-clipped rewards for TRAINING (DQN-family convention); evaluation and
    reward curves report raw game scores via info["raw_reward"].
  * Final-policy evaluation: 32 episodes at epsilon = 0.001, raw scores written
    to <run_dir>/eval_scores.csv and eval_summary.json.
  * Every game of the official 26-game suite has its own experiment dir
    (published SimPLe / OTRainbow / CURL / DrQ / SPR / MuZero / EfficientZero
    numbers exist for all of them); Enduro is NOT in the suite — it reports
    against random/human only and never enters the published-methods aggregate.
  * Whether a game's runs enter the aggregate-HNS comparison is selected in
    its config: experiment.include_in_aggregate (see aggregate_games()).

One (arm, seed, game) training per subprocess. Arms mirror the classical
launcher: the shared "baseline" (fhr_weight 0) plus one "exp<N>" arm per entry
of each game config's experiment.fhr_experiments (all entries by default).
The agent recipe is selected by experiment.agent_class in each game config:
"fhrdqn" (default: plain double-DQN FHRDQNAgent + NatureCNN),
"efficient_rainbow" (EfficientRainbowAgent: IQN-dueling + n-step + DrQ
augmentation over NatureCNNEncoder) or "bbf" (BBFAgent: the BBF recipe —
Impala-CNN x4, EMA target, n-step/gamma annealing, shrink-and-perturb resets
— over ImpalaCNNEncoder). Each recipe keeps its OWN manifest family per game
dir (cached/fhrdqn100k_runs_manifest.json vs cached/effrainbow100k_... vs
cached/bbf100k_...), so the result viewer and the comparison notebook never
pool runs across recipes or with non-100k arms.

Run from anywhere (game dirs are resolved against this file's location):

    python experiments/src/run_fhrdqn_atari100k.py                    # ALL 27 games x arms x seeds
    python experiments/src/run_fhrdqn_atari100k.py --games seaquest   # one game
    python experiments/src/run_fhrdqn_atari100k.py --experiment 1     # baseline + exp1 only
    python experiments/src/run_fhrdqn_atari100k.py --force            # rerun exp arms
    python experiments/src/run_fhrdqn_atari100k.py --steps 2000 --eval-episodes 2   # smoke

Child mode (one training in this process; cwd must be the game dir):

    python ../../src/run_fhrdqn_atari100k.py --arm baseline --seed 0

Run at most one launcher at a time per game dir (manifest + shared baseline
race). The comparison notebook imports launch_all()/load_runs() from here.
"""
import argparse
import copy
import csv
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent    # experiments/src
SRC = SCRIPTS_DIR.parents[1] / "src"                      # repo library code
ATARI_DIR = SCRIPTS_DIR.parent / "atari"
# All 26 suite games + Enduro (outside the suite). "pacman" keeps its historic
# short key; every other key is the snake_cased ALE id stem.
GAME_DIRS = {
    "alien": "dqn_alien", "amidar": "dqn_amidar", "assault": "dqn_assault",
    "asterix": "dqn_asterix", "bank_heist": "dqn_bank_heist",
    "battle_zone": "dqn_battle_zone", "boxing": "dqn_boxing",
    "breakout": "dqn_breakout", "chopper_command": "dqn_chopper_command",
    "crazy_climber": "dqn_crazy_climber", "demon_attack": "dqn_demon_attack",
    "enduro": "dqn_enduro", "freeway": "dqn_freeway",
    "frostbite": "dqn_frostbite", "gopher": "dqn_gopher", "hero": "dqn_hero",
    "jamesbond": "dqn_jamesbond", "kangaroo": "dqn_kangaroo",
    "krull": "dqn_krull", "kung_fu_master": "dqn_kung_fu_master",
    "pacman": "dqn_pacman", "pong": "dqn_pong",
    "private_eye": "dqn_private_eye", "qbert": "dqn_qbert",
    "road_runner": "dqn_road_runner", "seaquest": "dqn_seaquest",
    "up_n_down": "dqn_up_n_down",
}
CONFIG = "config_fhrdqn_100k.yaml"
# manifest/log family per agent recipe (experiment.agent_class config key)
FAMILIES = {"fhrdqn": "fhrdqn100k", "efficient_rainbow": "effrainbow100k",
            "bbf": "bbf100k", "sac_discrete": "sacd100k"}
LOGS = "cached/logs"

# Same single-source-of-truth FHR override whitelist as the classical launcher.
from run_fhrdqn_seeds import FHR_PARAMS  # noqa: E402


def _game_dir(game):
    if game not in GAME_DIRS:
        raise ValueError(f"unknown game {game!r} — one of {sorted(GAME_DIRS)}")
    d = ATARI_DIR / GAME_DIRS[game]
    if not (d / CONFIG).exists():
        raise FileNotFoundError(f"{d / CONFIG} does not exist")
    return d


def run_one(arm, seed, steps=None, agent_overrides=None, name_tag=None,
            eval_episodes=None):
    """Child mode: one full Atari-100k training + final evaluation in this
    process (cwd = game dir). The baseline arm strips the FHR term
    (fhr_weight = 0.0, the single difference); a numbered-experiment arm
    applies its FHR set on top of the config's agent block. Returns the run
    directory."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    # TF32 tensor-core matmuls/convs for these runs: the reference BBF (JAX)
    # trains with TF32-by-default on this hardware class; PyTorch's fp32
    # matmul default would make the 15488x2048 head layers the bottleneck.
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.compile (agent.torch_compile) needs CPython headers to build its
    # CUDA shim; this box has no root, so user-extracted ones stand in when
    # the system -dev package is absent (see ~/.local/pyheaders).
    _hdr = pathlib.Path.home() / ".local/pyheaders/extracted/usr/include"
    if _hdr.is_dir() and not pathlib.Path("/usr/include/python3.12/Python.h").exists():
        os.environ["CPATH"] = os.pathsep.join(
            [str(_hdr), str(_hdr / "python3.12")]
            + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))
    from experiment import load_config, build_env, build_agent, train, make_run_logger
    from agents.fhrdqn_agent import FHRDQNAgent
    from agents.efficient_rainbow_agent import EfficientRainbowAgent
    from agents.bbf_agent import BBFAgent
    from agents.sac_discrete_agent import SACDiscreteAgent
    from agents.atari_networks import (NatureCNN, NatureCNNEncoder,
                                       ImpalaCNNEncoder)
    from training import evaluate_policy_atari
    from analysis.atari100k import game_key, hns, REFERENCE

    cfg = load_config(CONFIG, seed=seed)          # seeds torch / numpy / random
    if arm == "baseline":
        cfg["agent"]["fhr_weight"] = 0.0
    elif agent_overrides:
        cfg["agent"].update(agent_overrides)
    if name_tag:
        cfg["experiment"]["name"] += f"_{name_tag}"
    elif arm == "baseline":
        cfg["experiment"]["name"] += "_baseline"  # shared-baseline naming
    cfg["experiment"]["name"] += f"_seed{seed}"
    if steps is not None:                         # smoke-test override
        cfg["training"]["max_env_steps"] = steps

    env = build_env(cfg)
    obs_shape = env.observation_space.shape       # (frame_stack, 84, 84)
    agent_class = cfg["experiment"].get("agent_class", "fhrdqn")
    if agent_class == "bbf":
        # BBF recipe: Impala-CNN(width network.width_scale) encoder only —
        # the agent wraps it with the IQN-dueling head (agent.head_hidden)
        agent = build_agent(cfg, env, ImpalaCNNEncoder,
                            {"in_channels": obs_shape[0],
                             "width_scale": cfg["network"].get("width_scale", 4)},
                            agent_cls=BBFAgent)
    elif agent_class == "efficient_rainbow":
        # encoder only — the agent wraps it with the IQN-dueling head itself
        # (head width comes from agent.head_hidden, not network.fc_hidden)
        agent = build_agent(cfg, env, NatureCNNEncoder,
                            {"in_channels": obs_shape[0]},
                            agent_cls=EfficientRainbowAgent)
    elif agent_class == "sac_discrete":
        # encoder only — the agent builds the categorical actor + twin
        # critic heads itself (width from agent.head_hidden)
        agent = build_agent(cfg, env, NatureCNNEncoder,
                            {"in_channels": obs_shape[0]},
                            agent_cls=SACDiscreteAgent)
    elif agent_class == "fhrdqn":
        nn_extra_kwargs = {"in_channels": obs_shape[0],
                           "n_actions": env.action_space.n,
                           "fc_hidden": cfg["network"]["fc_hidden"]}
        agent = build_agent(cfg, env, NatureCNN, nn_extra_kwargs,
                            agent_cls=FHRDQNAgent)
    else:
        raise ValueError(f"unknown experiment.agent_class {agent_class!r} — "
                         f"one of {sorted(FAMILIES)}")
    logger = make_run_logger(cfg, config_path=CONFIG, base_dir="cached")
    train(cfg, agent, env, run_logger=logger)

    # Final-policy evaluation on a FRESH env (the training env may sit in a
    # hijacked post-analysis state); raw scores, protocol in the config.
    # FULL-GAME episodes: the published Atari-100k numbers (random/human and
    # every method in src/analysis/atari100k.py) score complete games across
    # all lives, so the eval env must never terminate on life loss — no
    # matter how the TRAINING env treats lives.
    eval_cfg = dict(cfg.get("evaluation") or {})
    if eval_episodes is not None:                 # smoke-test override
        eval_cfg["episodes"] = eval_episodes
    eval_env_cfg = copy.deepcopy(cfg)
    eval_env_cfg["environment"]["atari"]["terminal_on_life_loss"] = False
    eval_env_cfg["environment"]["atari"]["episodic_life"] = False
    eval_env = build_env(eval_env_cfg)
    if hasattr(agent, "sample_actions"):
        # SAC-family eval protocol: argmax pi with prob 1 - epsilon (the
        # stochastic policy is training-time exploration only)
        agent.sample_actions = False
    scores = evaluate_policy_atari(agent, eval_env, **eval_cfg)
    eval_env.close()

    run_dir = pathlib.Path(logger.dir)
    with open(run_dir / "eval_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "score"])
        w.writerows(enumerate(scores))
    game = game_key(cfg["environment"]["name"])
    summary = {"game": game, "arm": arm, "name_tag": name_tag, "seed": seed,
               "episodes": len(scores),
               "epsilon": eval_cfg.get("epsilon", 0.001),
               "mean": float(np.mean(scores)), "median": float(np.median(scores)),
               "std": float(np.std(scores)),
               "min": float(np.min(scores)), "max": float(np.max(scores)),
               "hns_mean": hns(float(np.mean(scores)), game)
                           if game in REFERENCE else None}
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"EVAL mean {summary['mean']:.1f} over {len(scores)} episodes "
          f"(HNS {summary['hns_mean']})")
    return logger.dir


def _config_seeds(game_dir):
    with open(game_dir / CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]])


def _family(game_dir):
    """Manifest/log family of a game dir, from its config's agent_class."""
    with open(game_dir / CONFIG) as f:
        agent_class = yaml.safe_load(f)["experiment"].get("agent_class", "fhrdqn")
    if agent_class not in FAMILIES:
        raise ValueError(f"unknown experiment.agent_class {agent_class!r} in "
                         f"{game_dir / CONFIG} — one of {sorted(FAMILIES)}")
    return FAMILIES[agent_class]


def _manifest_rel(game_dir):
    return f"cached/{_family(game_dir)}_runs_manifest.json"


def config_game_key(game):
    """Benchmark game key from a game's config: 'pacman' -> 'MsPacman'."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.atari100k import game_key
    with open(_game_dir(game) / CONFIG) as f:
        return game_key(yaml.safe_load(f)["environment"]["name"])


def aggregate_games(games=None):
    """The games whose config opts into the aggregate-HNS comparison
    (experiment.include_in_aggregate: true). Games outside the official
    26-game suite (no published baselines, e.g. Enduro) are excluded no
    matter what their flag says — there is nothing to compare them against."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.atari100k import BASELINES
    selected = []
    for game in (sorted(GAME_DIRS) if games is None else games):
        with open(_game_dir(game) / CONFIG) as f:
            cfg = yaml.safe_load(f)
        if (cfg["experiment"].get("include_in_aggregate", False)
                and BASELINES.get(config_game_key(game))):
            selected.append(game)
    return selected


def _load_manifest(game_dir):
    path = game_dir / _manifest_rel(game_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": [], "runs": {}}


def _run_recorded(game_dir, manifest, arm, seed):
    rel = manifest["runs"].get(arm, {}).get(str(seed))
    return rel is not None and (game_dir / rel / "eval_summary.json").exists()


def _experiment_overrides(game_dir, experiment):
    with open(game_dir / CONFIG) as f:
        cfg = yaml.safe_load(f)
    sweeps = cfg["experiment"].get("fhr_experiments") or {}
    overrides = sweeps.get(experiment, sweeps.get(str(experiment)))
    if overrides is None:
        raise ValueError(
            f"experiment {experiment} is not defined under "
            f"experiment.fhr_experiments in {game_dir / CONFIG} "
            f"(defined: {sorted(map(str, sweeps)) or 'none'})")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"experiment {experiment} must be a non-empty mapping "
                         f"of FHR params — got {overrides!r}")
    unknown = sorted(set(overrides) - set(FHR_PARAMS))
    if unknown:
        raise ValueError(f"experiment {experiment} overrides unknown FHR "
                         f"params {unknown} — allowed: {list(FHR_PARAMS)}")
    if float(overrides.get("fhr_weight", 1.0)) == 0.0:
        raise ValueError("fhr_weight 0 is the baseline arm, which every "
                         "launch already includes — drop it from the sweep")
    return overrides


def _config_experiments(game_dir):
    """All experiment numbers defined in the game config, sorted."""
    with open(game_dir / CONFIG) as f:
        cfg = yaml.safe_load(f)
    return sorted(cfg["experiment"].get("fhr_experiments") or {})


def _arm_specs(game_dir, experiments):
    """[(manifest/log key, extra child argv, agent overrides)]: the shared
    baseline + one exp<N> arm per experiment (default: every entry the game
    config defines). Overrides travel as JSON and are recorded in the manifest
    (the run dir's config.yaml copy does not reflect them)."""
    if experiments is None:
        experiments = _config_experiments(game_dir)
    if len(set(experiments)) != len(experiments):
        raise ValueError(f"duplicate experiment numbers: {experiments}")
    specs = [("baseline", ["--arm", "baseline"], {"fhr_weight": 0.0})]
    for n in experiments:
        overrides = _experiment_overrides(game_dir, n)
        specs.append((f"exp{n}", ["--arm", "fhr",
                                  "--agent-overrides", json.dumps(overrides),
                                  "--name-tag", f"exp{n}"], overrides))
    return specs


def launch_all(games=None, experiments=None, max_workers=3, force=False,
               steps=None, eval_episodes=None):
    """Fan out one subprocess per (game, arm, seed). Pairs already recorded in
    a game's manifest are skipped unless force=True (force never retrains a
    shared baseline when experiments are requested — retrain it deliberately
    with force and experiments=[]). Atari CNNs are much heavier than the
    classic-control MLPs: default fan-out is 3. Returns {game: manifest};
    raises if any run fails (successes are kept; relaunch resumes)."""
    if isinstance(games, str):
        games = [games]
    games = list(games or sorted(GAME_DIRS))
    if isinstance(experiments, int):
        experiments = [experiments]

    jobs, manifests, locks = [], {}, {}
    for game in games:
        gd = _game_dir(game)
        seeds = _config_seeds(gd)
        manifest = _load_manifest(gd)
        manifest["seeds"] = seeds
        specs = _arm_specs(gd, experiments)
        manifest["overrides"] = {**manifest.get("overrides", {}),
                                 **{k: ov for k, _, ov in specs if ov is not None}}
        manifests[game], locks[game] = manifest, threading.Lock()
        (gd / LOGS).mkdir(parents=True, exist_ok=True)
        explicit_exps = experiments is not None and len(experiments) > 0
        for key, extra, _ in specs:
            for s in seeds:
                if ((force and (key != "baseline" or not explicit_exps))
                        or not _run_recorded(gd, manifest, key, s)):
                    jobs.append((game, gd, key, extra, s))
        # persist seeds/overrides even when everything is already trained
        with open(gd / _manifest_rel(gd), "w") as f:
            json.dump(manifest, f, indent=2)

    total = sum(len(_arm_specs(_game_dir(g), experiments)) *
                len(_config_seeds(_game_dir(g))) for g in games)
    if total - len(jobs):
        print(f"skipping {total - len(jobs)} already-completed run(s)")
    if not jobs:
        return manifests

    child_env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1")
    failures = []
    flock = threading.Lock()

    def work(job):
        game, gd, key, extra, seed = job
        log_path = gd / LOGS / f"{_family(gd)}_{key}_seed{seed}.log"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               *extra, "--seed", str(seed), "--config", CONFIG]
        if steps is not None:
            cmd += ["--steps", str(steps)]
        if eval_episodes is not None:
            cmd += ["--eval-episodes", str(eval_episodes)]
        t0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=gd, env=child_env,
                                  stdout=lf, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        run_dir = None
        for line in reversed(log_path.read_text().splitlines()):
            if line.startswith("RUN_DIR="):
                run_dir = line.removeprefix("RUN_DIR=")
                break
        if proc.returncode != 0 or run_dir is None:
            print(f"[{game} {key} seed {seed}] FAILED after {mins:.1f} min — see {log_path}")
            with flock:
                failures.append((game, key, seed, str(log_path)))
            return
        rel = os.path.relpath(os.path.join(gd, run_dir), gd)
        with locks[game]:
            manifests[game]["runs"].setdefault(key, {})[str(seed)] = rel
            with open(gd / _manifest_rel(gd), "w") as f:
                json.dump(manifests[game], f, indent=2)
        print(f"[{game} {key} seed {seed}] done in {mins:.1f} min -> {rel}")

    print(f"launching {len(jobs)} run(s), {min(max_workers, len(jobs))} at a "
          f"time: {[(g, k, s) for g, _, k, _, s in jobs]}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, jobs))
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed (completed runs are "
                           f"kept; relaunch to retry): {failures}")
    return manifests


def load_runs(game, arm):
    """Manifest -> [{seed, cfg, rewards, steps, eval_scores, eval_summary,
    agent_overrides, run_dir}] in config-seed order. rewards/steps come from
    rewards.csv (TRAINING returns: raw game score per training episode);
    eval_scores/eval_summary are the separate final-policy evaluation. cfg is
    the run dir's config copy — it reflects neither the baseline's
    fhr_weight=0 nor an experiment's overrides (see agent_overrides)."""
    gd = _game_dir(game)
    manifest = _load_manifest(gd)
    if not manifest["seeds"]:
        raise RuntimeError(f"no runs recorded in {gd / _manifest_rel(gd)} — "
                           "launch_all() first")
    runs = []
    for seed in manifest["seeds"]:
        rel = manifest["runs"].get(arm, {}).get(str(seed))
        if rel is None or not (gd / rel / "eval_summary.json").exists():
            raise RuntimeError(f"no completed {game}/{arm} run for seed {seed} "
                               f"— launch_all() it first ({LOGS}/ has failures)")
        run_dir = gd / rel
        table = np.loadtxt(run_dir / "rewards.csv", delimiter=",", skiprows=1,
                           ndmin=2)
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        with open(run_dir / "eval_summary.json") as f:
            eval_summary = json.load(f)
        eval_scores = np.loadtxt(run_dir / "eval_scores.csv", delimiter=",",
                                 skiprows=1, ndmin=2)[:, 1]
        runs.append({"seed": seed, "cfg": cfg, "rewards": table[:, 1],
                     "steps": table[:, 2] if table.shape[1] > 2 else None,
                     "eval_scores": eval_scores, "eval_summary": eval_summary,
                     "agent_overrides": manifest.get("overrides", {}).get(arm),
                     "run_dir": run_dir})
    return runs


def arms_of(game):
    """Arm keys recorded in a game's manifest, baseline first then exp<N> asc."""
    manifest = _load_manifest(_game_dir(game))
    keys = list(manifest["runs"])
    order = lambda k: (0, 0) if k == "baseline" else \
        (1, int(k[3:])) if k.startswith("exp") and k[3:].isdigit() else (2, 0)
    return sorted(keys, key=order)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("baseline", "fhr"),
                    help="child mode: train one (arm, seed) with cwd = game dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--agent-overrides", type=json.loads, default=None)
    ap.add_argument("--name-tag", default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="override training.max_env_steps (smoke tests)")
    ap.add_argument("--eval-episodes", type=int, default=None,
                    help="override evaluation.episodes (smoke tests)")
    ap.add_argument("--games", nargs="*", default=None,
                    help=f"launcher mode: subset of {sorted(GAME_DIRS)}")
    ap.add_argument("--experiment", type=int, nargs="*", default=None,
                    help="numbered fhr_experiments entries (default: all defined)")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=CONFIG,
                    help="per-game config filename (default: %(default)s). "
                         "Lets one game dir host several recipes, each config "
                         "carrying its own agent_class and thus its own "
                         "manifest family — e.g. config_effrainbow_100k.yaml")
    args = ap.parse_args()
    globals()["CONFIG"] = args.config             # every helper reads the global

    if args.arm is not None:                      # child mode
        if args.seed is None:
            ap.error("--arm requires --seed")
        run_dir = run_one(args.arm, args.seed, steps=args.steps,
                          agent_overrides=args.agent_overrides,
                          name_tag=args.name_tag,
                          eval_episodes=args.eval_episodes)
        print(f"RUN_DIR={run_dir}")
        return
    launch_all(games=args.games, experiments=args.experiment,
               max_workers=args.max_workers, force=args.force,
               steps=args.steps, eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
