"""Atari-100k benchmark metadata and human-normalised scoring.

The benchmark (introduced by SimPLe, Kaiser et al. 2020; the standard venue for
sample-efficient Atari: DER, OTRainbow, CURL, DrQ, SPR, MuZero-100k,
EfficientZero, BBF) trains for 100,000 agent-environment interactions (400k
frames at frameskip 4) and reports the final policy's raw game score, usually
normalised against human and random reference play:

    HNS = (score_agent - score_random) / (score_human - score_random)

Reference and baseline numbers here are quoted VERBATIM from
Ye et al. 2021, "Mastering Atari Games with Limited Data" (EfficientZero,
arXiv:2111.00210, Table 1), whose Random/Human columns match the standard
reference table (Wang et al. 2016, arXiv:1511.06581, Table 2 — 30 no-op
starts). Enduro's random/human values come from that same Wang et al. table.

The official benchmark suite is a fixed 26-game subset; all 26 are tabulated
below (BENCHMARK_GAMES). **Enduro is NOT in the suite** — it has no published
Atari-100k baselines and can only be reported against random/human; it must
never enter an aggregate compared against published methods. Any aggregate
over fewer than the 26 games is a subset, not "the Atari-100k benchmark" —
label figures with the games it covers. Which games this repo's experiments
feed into the aggregate is selected per game config
(experiment.include_in_aggregate; experiments/src/run_fhrdqn_atari100k.py
aggregate_games()).
"""

ATARI100K_ENV_STEPS = 100_000     # agent-env interactions (x4 frames)
EVAL_EPISODES = 32                # EfficientZero's evaluation protocol
EVAL_EPSILON = 0.001              # DER/SPR near-greedy evaluation epsilon

# (random, human) reference scores, 30 no-op starts regime. Keys are
# game_key() outputs, i.e. the ALE id stem ("ALE/BankHeist-v5" -> "BankHeist").
REFERENCE = {
    "Alien":          {"random": 227.8,   "human": 7127.7},
    "Amidar":         {"random": 5.8,     "human": 1719.5},
    "Assault":        {"random": 222.4,   "human": 742.0},
    "Asterix":        {"random": 210.0,   "human": 8503.3},
    "BankHeist":      {"random": 14.2,    "human": 753.1},
    "BattleZone":     {"random": 2360.0,  "human": 37187.5},
    "Boxing":         {"random": 0.1,     "human": 12.1},
    "Breakout":       {"random": 1.7,     "human": 30.5},
    "ChopperCommand": {"random": 811.0,   "human": 7387.8},
    "CrazyClimber":   {"random": 10780.5, "human": 35829.4},
    "DemonAttack":    {"random": 152.1,   "human": 1971.0},
    "Freeway":        {"random": 0.0,     "human": 29.6},
    "Frostbite":      {"random": 65.2,    "human": 4334.7},
    "Gopher":         {"random": 257.6,   "human": 2412.5},
    "Hero":           {"random": 1027.0,  "human": 30826.4},
    "Jamesbond":      {"random": 29.0,    "human": 302.8},
    "Kangaroo":       {"random": 52.0,    "human": 3035.0},
    "Krull":          {"random": 1598.0,  "human": 2665.5},
    "KungFuMaster":   {"random": 258.5,   "human": 22736.3},
    "MsPacman":       {"random": 307.3,   "human": 6951.6},
    "Pong":           {"random": -20.7,   "human": 14.6},
    "PrivateEye":     {"random": 24.9,    "human": 69571.3},
    "Qbert":          {"random": 163.9,   "human": 13455.0},
    "RoadRunner":     {"random": 11.5,    "human": 7845.0},
    "Seaquest":       {"random": 68.4,    "human": 42054.7},
    "UpNDown":        {"random": 533.4,   "human": 11693.2},
    "Enduro":         {"random": 0.0,     "human": 860.5},   # not in the suite
}

# Published Atari-100k final scores (raw game score), EfficientZero Table 1 —
# the full 26-game suite. Enduro is outside the suite: no published numbers.
_METHOD_ORDER = ("SimPLe", "OTRainbow", "CURL", "DrQ", "SPR", "MuZero",
                 "EfficientZero")
_TABLE1 = {
    #                  SimPLe  OTRainbow    CURL      DrQ      SPR   MuZero  EffZero
    "Alien":          (616.9,     824.7,   558.2,   771.2,   801.5,   530.0,   808.5),
    "Amidar":         (88.0,       82.8,   142.1,   102.8,   176.3,    38.8,   148.6),
    "Assault":        (527.2,     351.9,   600.6,   452.4,   571.0,   500.1,  1263.1),
    "Asterix":        (1128.3,    628.5,   734.5,   603.5,   977.8,  1734.0, 25557.8),
    "BankHeist":      (34.2,      182.1,   131.6,   168.9,   380.9,   192.5,   351.0),
    "BattleZone":     (5184.4,   4060.6, 14870.0, 12954.0, 16651.0,  7687.5, 13871.2),
    "Boxing":         (9.1,         2.5,     1.2,     6.0,    35.8,    15.1,    52.7),
    "Breakout":       (16.4,        9.8,     4.9,    16.1,    17.1,    48.0,   414.1),
    "ChopperCommand": (1246.9,   1033.3,  1058.5,   780.3,   974.8,  1350.0,  1117.3),
    "CrazyClimber":   (62583.6, 21327.8, 12146.5, 20516.5, 42923.6, 56937.0, 83940.2),
    "DemonAttack":    (208.1,     711.8,   817.6,  1113.4,   545.2,  3527.0, 13003.9),
    "Freeway":        (20.3,       25.0,    26.7,     9.8,    24.4,    21.8,    21.8),
    "Frostbite":      (254.7,     231.6,  1181.3,   331.1,  1821.5,   255.0,   296.3),
    "Gopher":         (771.0,     778.0,   669.3,   636.3,   715.2,  1256.0,  3260.3),
    "Hero":           (2656.6,   6458.8,  6279.3,  3736.3,  7019.2,  3095.0,  9315.9),
    "Jamesbond":      (125.3,     112.3,   471.0,   236.0,   365.4,    87.5,   517.0),
    "Kangaroo":       (323.1,     605.4,   872.5,   940.6,  3276.4,    62.5,   724.1),
    "Krull":          (4539.9,   3277.9,  4229.6,  4018.1,  3688.9,  4890.8,  5663.3),
    "KungFuMaster":   (17257.2,  5722.2, 14307.8,  9111.0, 13192.7, 18813.0, 30944.8),
    "MsPacman":       (1480.0,    941.9,  1465.5,   960.5,  1313.2,  1265.6,  1281.2),
    "Pong":           (12.8,        1.3,   -16.5,    -8.5,    -5.9,    -6.7,    20.1),
    "PrivateEye":     (58.3,      100.0,   218.4,   -13.6,   124.0,    56.3,    96.7),
    "Qbert":          (1288.8,    509.3,  1042.4,   854.4,   669.1,  3952.0, 13781.9),
    "RoadRunner":     (5640.6,   2696.7,  5661.0,  8895.1, 14220.5,  2500.0, 17751.3),
    "Seaquest":       (683.3,     286.9,   384.5,   301.2,   583.1,   208.0,  1100.2),
    "UpNDown":        (3350.0,   2847.6,  2955.2,  3180.8, 28138.5,  2896.9, 17264.2),
}
BASELINES = {g: dict(zip(_METHOD_ORDER, row)) for g, row in _TABLE1.items()}
BASELINES["Enduro"] = {}

# The games both this repo and the published tables cover — the only games an
# aggregate against published methods may legitimately include.
BENCHMARK_GAMES = tuple(g for g, b in BASELINES.items() if b)


def game_key(env_name: str) -> str:
    """Canonical game key from a Gymnasium ALE id: 'ALE/MsPacman-v5' -> 'MsPacman'."""
    return env_name.split("/")[-1].split("-")[0]


def hns(score: float, game: str) -> float:
    """Human-normalised score: 0 = random play, 1 = human reference."""
    ref = REFERENCE[game]
    return (score - ref["random"]) / (ref["human"] - ref["random"])


def aggregate_hns(scores_by_game: dict) -> dict:
    """{'mean': ..., 'median': ...} of HNS over the given {game: raw_score}.
    With so few games the median is of limited value — report both, and say in
    the figure how many games the aggregate covers."""
    vals = sorted(hns(s, g) for g, s in scores_by_game.items())
    n = len(vals)
    if not n:
        return {"mean": float("nan"), "median": float("nan")}
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"mean": sum(vals) / n, "median": med}
