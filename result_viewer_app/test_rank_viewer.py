"""Tests for the rank viewer server (stdlib only, like the server itself).

    python result_viewer_app/test_rank_viewer.py

Covers the on-disk data contracts the viewer depends on — the run-dir scan,
the launcher manifest (cached/*manifest*.json), summary payloads — and the
HTTP layer including path-traversal refusal. The manifest parser is fed
malformed inputs on purpose: the viewer must degrade, never crash, when the
experiment code that writes these files changes.
"""
import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rank_viewer  # noqa: E402


MANIFEST = {
    "seeds": [44, 66],
    "runs": {
        "baseline": {"44": "cached/runs/mc_fhrdqn_baseline_seed44_20260813-175543"},
        "exp2": {"44": "cached/runs/mc_fhrdqn_exp2_seed44_20260814-010101"},
    },
    "overrides": {
        "baseline": {"fhr_weight": 0.0},
        "exp2": {"fhr_weight": 0.2},
    },
}

CONFIG_YAML = """\
experiment:
  name: mc_fhrdqn
  seed: 44
environment:
  name: MountainCar-v0
agent:
  fhr_weight: 0.5
  fhr_order: 2
training:
  no_episodes: 5
"""


def make_run(run_dir: pathlib.Path, *, steps_col=True, config=True):
    run_dir.mkdir(parents=True)
    if steps_col:
        (run_dir / "rewards.csv").write_text(
            "episode,reward,steps\n0,-200.0,200\n1,-180.0,180\n")
    else:
        (run_dir / "rewards.csv").write_text("episode,reward\n0,-200.0\n1,-180.0\n")
    if config:
        (run_dir / "config.yaml").write_text(CONFIG_YAML)


class TreeCase(unittest.TestCase):
    """Shared temp tree: one manifest-tracked experiment (with a stale run the
    manifest does not list) and one plain experiment without a manifest."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.exp = "classical_control/dqn_mountaincar"
        exp_dir = self.root / self.exp
        make_run(exp_dir / "cached/runs/mc_fhrdqn_baseline_seed44_20260813-175543")
        make_run(exp_dir / "cached/runs/mc_fhrdqn_exp2_seed44_20260814-010101")
        make_run(exp_dir / "cached/runs/mc_fhrdqn_stale_seed44_20260101-000000")
        (exp_dir / "cached/fhrdqn_runs_manifest.json").write_text(
            json.dumps(MANIFEST))
        make_run(self.root / "other_exp/runs/plain_run_s0_20260801-120000",
                 steps_col=False)

    def tearDown(self):
        self._tmp.cleanup()


class TestScanManifests(TreeCase):
    def test_arms_overrides_and_run_index(self):
        m = rank_viewer.scan_manifests(self.root)
        self.assertEqual(set(m), {self.exp})
        man = m[self.exp]
        self.assertEqual(set(man["families"]), {"fhrdqn_runs"})
        fam = man["families"]["fhrdqn_runs"]
        self.assertEqual(set(fam["arms"]), {"baseline", "exp2"})
        # rel paths reduce to run-dir names, seeds normalise to strings
        self.assertEqual(fam["arms"]["exp2"],
                         {"44": "mc_fhrdqn_exp2_seed44_20260814-010101"})
        self.assertEqual(fam["overrides"]["exp2"], {"fhr_weight": 0.2})
        self.assertEqual(
            man["by_run"]["mc_fhrdqn_baseline_seed44_20260813-175543"],
            {"arm": "baseline", "seed": "44", "family": "fhrdqn_runs"})

    def test_archived_family_never_merges_with_current(self):
        # an archived manifest (fhrdqn_runs_manifest.old-lambda0.5.json) also
        # has "baseline"/"fhr" arms — pooling them with the current family
        # would average runs from different experiment recipes
        exp_dir = self.root / self.exp
        (exp_dir / "cached/fhrdqn_runs_manifest.old-lambda0.5.json").write_text(
            json.dumps({"runs": {"baseline": {
                "21": "cached/runs/mc_old_baseline_seed21"}}}))
        man = rank_viewer.scan_manifests(self.root)[self.exp]
        self.assertEqual(set(man["families"]),
                         {"fhrdqn_runs", "fhrdqn_runs.old-lambda0.5"})
        old = man["families"]["fhrdqn_runs.old-lambda0.5"]
        self.assertEqual(old["arms"]["baseline"],
                         {"21": "mc_old_baseline_seed21"})
        # the current family's baseline is untouched by the archived one
        cur = man["families"]["fhrdqn_runs"]
        self.assertEqual(set(cur["arms"]["baseline"]), {"44"})
        self.assertEqual(man["by_run"]["mc_old_baseline_seed21"]["family"],
                         "fhrdqn_runs.old-lambda0.5")

    def test_malformed_manifests_are_skipped_field_by_field(self):
        exp_dir = self.root / self.exp
        (exp_dir / "cached/broken_manifest.json").write_text("{ not json")
        (exp_dir / "cached/odd_manifest.json").write_text(json.dumps({
            "runs": {"weird": "not-a-dict",
                     "ok": {"7": "cached/runs/somewhere"}},
            "overrides": "not-a-dict",
        }))
        man = rank_viewer.scan_manifests(self.root)[self.exp]
        self.assertIn("odd", man["families"])       # valid parts of odd file
        odd = man["families"]["odd"]
        self.assertIn("ok", odd["arms"])
        self.assertNotIn("weird", odd["arms"])      # non-dict arm ignored
        self.assertNotIn("broken", man["families"])  # unparsable file skipped
        fam = man["families"]["fhrdqn_runs"]        # real manifest unaffected
        self.assertEqual(fam["overrides"].get("exp2"), {"fhr_weight": 0.2})

    def test_no_manifest_dir_yields_nothing(self):
        m = rank_viewer.scan_manifests(self.root / "other_exp")
        self.assertEqual(m, {})


class TestScanRuns(TreeCase):
    def test_annotation(self):
        manifests = rank_viewer.scan_manifests(self.root)
        rows = {r["run"]: r for r in rank_viewer.scan_runs(self.root, manifests)}
        self.assertEqual(len(rows), 4)
        base = rows["mc_fhrdqn_baseline_seed44_20260813-175543"]
        self.assertEqual((base["arm"], base["seed"], base["tracked"]),
                         ("baseline", "44", True))
        self.assertEqual(base["family"], "fhrdqn_runs")
        stale = rows["mc_fhrdqn_stale_seed44_20260101-000000"]
        self.assertEqual(stale.get("tracked"), False)
        self.assertNotIn("arm", stale)
        plain = rows["plain_run_s0_20260801-120000"]
        self.assertNotIn("tracked", plain)        # exp has no manifest at all

    def test_without_manifests_matches_legacy_shape(self):
        rows = rank_viewer.scan_runs(self.root)
        self.assertTrue(all("arm" not in r and "tracked" not in r for r in rows))


class TestSummary(TreeCase):
    def test_summary_includes_config_and_logged_steps(self):
        d = (self.root / self.exp
             / "cached/runs/mc_fhrdqn_baseline_seed44_20260813-175543")
        s = rank_viewer.load_summary(d)
        self.assertEqual(s["steps_source"], "logged")
        self.assertIn("MountainCar-v0", s["config"])
        self.assertEqual(s["rewards"][0], [0, -200.0, 200.0])

    def test_derived_steps_for_legacy_two_column_rewards(self):
        d = self.root / self.exp / "cached/runs/legacy"
        make_run(d, steps_col=False)
        s = rank_viewer.load_summary(d)
        self.assertEqual(s["steps_source"], "derived")
        self.assertEqual(s["rewards"][0][2], 200)


SWEEP_CSV = (
    "episode,matrix,rollout,seed,sub_len,eff_rank,stable_rank,spikiness,"
    "n_rows,n_cols,nnz_rows,nnz_cols,row_coherence,col_coherence,"
    "row_lev_min,row_lev_max,col_lev_min,col_lev_max,sv_01,sv_02,sv_03\n"
    "0,Hankel Q,0,52,5,2,1.01,1.09,3,3,3,3,1.5,1.5,0.24,0.5,0.24,0.5,"
    "0.266909,0.0257733,nan\n"
    "50,Hankel Q,0,52,5,8,3.86,1.19,8,8,8,8,1.0,1.0,0.12,0.12,0.12,0.12,"
    "0.331891,0.317139,0.28996\n")


class TestSpectra(TreeCase):
    def _run_dir(self):
        return (self.root / self.exp
                / "cached/runs/mc_fhrdqn_baseline_seed44_20260813-175543")

    def test_spectra_rows_with_nan_padding_stripped(self):
        (self._run_dir() / "hankel_sweep.csv").write_text(SWEEP_CSV)
        s = rank_viewer.load_spectra(self._run_dir())
        self.assertEqual(len(s["rows"]), 2)
        ep, matrix, rollout, sub_len, nr, nc, sv = s["rows"][0]
        self.assertEqual((ep, matrix, rollout, sub_len, nr, nc),
                         (0, "Hankel Q", 0, 5, 3, 3))
        self.assertEqual(sv, [0.266909, 0.0257733])   # trailing nan dropped
        self.assertEqual(len(s["rows"][1][6]), 3)
        # round-trips through strict JSON (no NaN tokens)
        json.loads(json.dumps(s))

    def test_legacy_sweep_without_sv_columns(self):
        (self._run_dir() / "hankel_sweep.csv").write_text(
            "episode,matrix,rollout,seed,sub_len,eff_rank\n"
            "0,Hankel Q,0,52,5,2\n")
        self.assertIsNone(rank_viewer.load_spectra(self._run_dir())["rows"])

    def test_missing_sweep(self):
        self.assertIsNone(rank_viewer.load_spectra(self._run_dir())["rows"])


class TestHTTP(TreeCase):
    def setUp(self):
        super().setUp()
        handler = rank_viewer.make_handler(self.root)
        self.server = rank_viewer.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        super().tearDown()

    def get(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_api_runs_payload_shape(self):
        code, body = self.get("/api/runs")
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertIn("runs", payload)
        self.assertIn("manifests", payload)
        self.assertIn(self.exp, payload["manifests"])
        man = payload["manifests"][self.exp]
        self.assertEqual(set(man), {"families"})    # by_run stays private
        fam = man["families"]["fhrdqn_runs"]
        self.assertEqual(set(fam), {"arms", "overrides"})
        tracked = [r for r in payload["runs"] if r.get("tracked")]
        self.assertEqual({r["arm"] for r in tracked}, {"baseline", "exp2"})

    def test_summary_endpoint(self):
        code, body = self.get(
            f"/api/summary/{self.exp}/mc_fhrdqn_exp2_seed44_20260814-010101")
        self.assertEqual(code, 200)
        s = json.loads(body)
        self.assertIn("config", s)
        self.assertEqual(s["steps_source"], "logged")

    def test_unknown_run_404(self):
        code, body = self.get(f"/api/summary/{self.exp}/nope")
        self.assertEqual(code, 404)

    def test_spectra_endpoint(self):
        run = "mc_fhrdqn_exp2_seed44_20260814-010101"
        (self.root / self.exp / "cached/runs" / run
         / "hankel_sweep.csv").write_text(SWEEP_CSV)
        code, body = self.get(f"/api/spectra/{self.exp}/{run}")
        self.assertEqual(code, 200)
        s = json.loads(body)
        self.assertEqual(len(s["rows"]), 2)
        self.assertEqual(s["rows"][1][6], [0.331891, 0.317139, 0.28996])
        code, _ = self.get(f"/api/spectra/{self.exp}/nope")
        self.assertEqual(code, 404)

    def test_path_traversal_refused(self):
        outside = self.root.parent / "secret.csv"
        outside.write_text("top,secret\n")
        try:
            for path in ("/csv/../../secret.csv",
                         "/csv/%2e%2e/%2e%2e/secret.csv",
                         f"/csv/{self.exp}/mc_fhrdqn_exp2_seed44_20260814-010101/"
                         "%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/secret.csv"):
                code, body = self.get(path)
                self.assertEqual(code, 404, f"{path} must not resolve")
                self.assertNotIn(b"top,secret", body)
        finally:
            outside.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
