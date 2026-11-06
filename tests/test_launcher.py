"""Offline launcher/delivery tests; no Docker or provider generation."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("run_demo", Path(__file__).resolve().parents[1] / "runtime/app/repoagent/legacy_launcher.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.harness = self.root / "harness"
        (self.harness / "app/corecoder_mini").mkdir(parents=True)
        (self.harness / "app/corecoder_mini/supervisor.py").touch()
        self.profile = {"service": "DeepSeek Open Platform", "provider": "openai",
                        "base_url": "https://api.deepseek.com", "default_model": "fixture-model",
                        "api_key_env": "DEMO_TEST_KEY", "credential_file": str(self.root / "private-key.txt")}
        (self.root / "private-key.txt").write_text("sk-fake-unit-test-only", encoding="utf-8")
        demo.write_json(self.harness / "API_PROVIDER_CURRENT.json", self.profile)
        self.packet = self.root / "packet"
        self.target = self.root / "new-project"

    def prepare(self):
        return demo.prepare("实现一个计算器", self.target, self.packet, self.harness, image="fixture-image")

    def completed_run(self):
        self.prepare()
        # Reuse the actual final artifact routines, not a second patch implementation.
        meta = demo.read_json(self.packet / "demo.json")
        meta["harness"] = str(demo.DEFAULT_HARNESS)
        demo.write_json(self.packet / "demo.json", meta)
        run = self.packet / "run"
        (run / "baseline").mkdir(parents=True)
        (run / "workspace").mkdir()
        (run / "workspace/main.py").write_text("print('fixture')\n", encoding="utf-8")
        io = demo.artifact_io(demo.DEFAULT_HARNESS)
        (run / "model.patch").write_bytes(io.filesystem_patch(run / "baseline", run / "workspace"))
        demo.write_json(run / "frozen.json", {"baseline_sha256": io.tree_hashes(run / "baseline")})
        demo.write_json(run / "verification.json", {"ok": True, "resolved": True, "patch_applied": True,
                        "patch_sha256": demo.sha(run / "model.patch")})
        demo.write_json(run / "result.json", {"autonomous_complete": True, "model_submitted": True,
                        "verification_ok": True, "artifact_correct": True,
                        "patch_sha256": demo.sha(run / "model.patch"),
                        "evidence_sha256": demo.sha(run / "verification.json")})
        return run

    def test_preparation_is_offline_and_never_copies_key(self):
        with patch.object(demo, "credential_environment", side_effect=AssertionError("No key access")), \
             patch.object(demo.subprocess, "Popen", side_effect=AssertionError("No subprocess")):
            config = self.prepare()
        text = "".join(p.read_text(encoding="utf-8") for p in self.packet.glob("*.json"))
        self.assertNotIn("sk-fake-unit-test-only", text)
        self.assertEqual(config["settings"]["task_intent"], "implement")
        self.assertIn("todo_write", config["task"])
        self.assertFalse(self.target.exists())
        self.assertFalse((self.packet / "plan.json").exists())

    def test_existing_destination_rejected_without_overwrite(self):
        self.target.mkdir()
        sentinel = self.target / "existing.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse(self.packet.exists())

    def test_run_and_destination_cannot_be_nested(self):
        with self.assertRaises(ValueError):
            demo.prepare("new task", self.target, self.target / "run", self.harness, image="image")

    def test_bad_budget_rejected_before_creating_packet(self):
        with self.assertRaises(ValueError):
            demo.prepare("new task", self.target, self.packet, self.harness, image="image", dollars=float("nan"))
        self.assertFalse(self.packet.exists())

    def test_private_key_only_enters_child_environment(self):
        env = demo.credential_environment(self.profile, inherited={})
        self.assertEqual(env["DEMO_TEST_KEY"], "sk-fake-unit-test-only")
        self.assertNotIn("DEMO_TEST_KEY", __import__("os").environ)

    def test_model_plan_is_not_fabricated(self):
        self.prepare()
        self.assertFalse(demo.export_plan(self.packet)["model_plan_recorded"])
        state = self.packet / "run/agent/working-state.json"
        state.parent.mkdir(parents=True)
        original = {"intent": {"kind": "implement", "summary": "model fixture"},
                    "todos": [{"id": "t1", "content": "implement", "status": "pending"}], "revision": 1}
        demo.write_json(state, original)
        plan = demo.export_plan(self.packet)
        self.assertTrue(plan["model_plan_recorded"])
        self.assertEqual(plan["tasks"], original["todos"])
        self.assertFalse(plan["is_acceptance"])

    def test_publish_exact_verified_artifact(self):
        run = self.completed_run()
        (run / "workspace/__pycache__").mkdir()
        (run / "workspace/__pycache__/cache.pyc").write_bytes(b"junk")
        self.assertEqual(demo.publish(self.packet), self.target)
        self.assertEqual((self.target / "main.py").read_bytes(), (run / "workspace/main.py").read_bytes())
        self.assertFalse((self.target / "__pycache__").exists())
        self.assertTrue(demo.read_json(self.packet / "delivery.json")["published"])

    def test_rejected_result_never_publishes(self):
        run = self.completed_run()
        result = demo.read_json(run / "result.json")
        result["autonomous_complete"] = False
        demo.write_json(run / "result.json", result)
        with self.assertRaises(ValueError):
            demo.publish(self.packet)
        self.assertFalse(self.target.exists())

    def test_workspace_change_after_grading_rejected(self):
        run = self.completed_run()
        (run / "workspace/main.py").write_text("changed after validation")
        with self.assertRaises(ValueError):
            demo.publish(self.packet)
        self.assertFalse(self.target.exists())

    def test_evidence_change_rejected(self):
        run = self.completed_run()
        (run / "verification.json").write_text("{}")
        with self.assertRaises(ValueError):
            demo.publish(self.packet)
        self.assertFalse(self.target.exists())

    def test_destination_created_during_run_is_preserved(self):
        self.completed_run()
        self.target.mkdir()
        (self.target / "user.txt").write_text("keep")
        with self.assertRaises(ValueError):
            demo.publish(self.packet)
        self.assertEqual((self.target / "user.txt").read_text(), "keep")

    def test_prior_run_never_restarts_provider(self):
        self.prepare()
        (self.packet / "run").mkdir()
        with patch.object(demo, "credential_environment", side_effect=AssertionError("No key access")), \
             self.assertRaises(ValueError):
            demo.run_packet(self.packet)

    def test_run_uses_final_harness_once_and_publishes_after_exit(self):
        self.prepare()
        with patch.object(demo.subprocess, "Popen") as popen, \
             patch.object(demo, "publish", return_value=self.target) as publish:
            popen.return_value.wait.return_value = 0
            self.assertEqual(demo.run_packet(self.packet), 0)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("corecoder_mini.supervisor", command)
        self.assertEqual(popen.call_args.kwargs["cwd"], self.harness / "app")
        self.assertNotIn("sk-fake-unit-test-only", " ".join(command))
        self.assertEqual(popen.call_args.kwargs["env"]["DEMO_TEST_KEY"], "sk-fake-unit-test-only")
        publish.assert_called_once_with(self.packet)

    def test_run_failure_exports_plan_without_publishing(self):
        self.prepare()
        with patch.object(demo.subprocess, "Popen") as popen, patch.object(demo, "publish") as publish:
            popen.return_value.wait.return_value = 1
            self.assertEqual(demo.run_packet(self.packet), 1)
        publish.assert_not_called()
        self.assertFalse(demo.read_json(self.packet / "plan.json")["model_plan_recorded"])

    def test_failure_diagnostics_do_not_export_private_arguments(self):
        self.prepare()
        run = self.packet / "run"
        (run / "agent").mkdir(parents=True)
        demo.write_json(run / "result.json", {"checkpoint_exit": "RepeatedFormatError", "state": "rejected"})
        demo.write_json(run / "agent/trajectory.json", {"messages": [{"extra": {
            "interrupt_type": "FormatError", "response": {"choices": [{"message": {"tool_calls": [{
                "function": {"name": "todo_write", "arguments": '{"private":"do-not-export"}}'}}]}}]}}}]})
        summary = demo.failure_summary(self.packet)
        self.assertEqual(summary["exit_status"], "RepeatedFormatError")
        self.assertEqual(summary["tool_argument_errors"][0]["category"], "Extra data")
        self.assertNotIn("do-not-export", (self.packet / "diagnostics.json").read_text())

    def test_packaged_demo_uses_its_own_relocatable_profile(self):
        packet = self.root / "real-demo-packet"
        config = demo.prepare("Public preparation fixture", self.target, packet,
                              demo.DEFAULT_HARNESS, image="fixture-image")
        meta = demo.read_json(packet / "demo.json")
        self.assertEqual(Path(meta["provider_profile"]), demo.DEFAULT_HARNESS / "API_PROVIDER_CURRENT.json")
        self.assertEqual(Path(meta["harness"]), demo.DEFAULT_HARNESS)
        self.assertEqual(config["api_key_env"], "DEEPSEEK_API_KEY")

    def test_interrupt_requests_stop_without_replay_or_publish(self):
        self.prepare()
        def interrupted(**kwargs):
            (self.packet / "run").mkdir()
            raise KeyboardInterrupt
        with patch.object(demo.subprocess, "Popen") as popen, patch.object(demo, "publish") as publish:
            popen.return_value.wait.side_effect = interrupted
            self.assertEqual(demo.run_packet(self.packet), 130)
        publish.assert_not_called()
        self.assertTrue((self.packet / "run/STOP").exists())
        popen.assert_called_once()

    def test_default_check_rejects_zero_and_failing_tests(self):
        # This executes our own public unit fixtures, never model-produced host code.
        tests = self.root / "tests"
        tests.mkdir()
        code = demo.DEFAULT_CHECK.removeprefix('python -c "')[:-1]
        def check():
            return subprocess.run([sys.executable, "-B", "-c", code], cwd=self.root,
                                  capture_output=True, timeout=10, check=False).returncode
        self.assertNotEqual(check(), 0)
        example = tests / "test_example.py"
        example.write_text("import unittest\nclass Example(unittest.TestCase):\n def test_example(self): self.assertTrue(False)\n")
        self.assertNotEqual(check(), 0)
        example.write_text("import unittest\nclass Example(unittest.TestCase):\n def test_example(self): self.assertTrue(True)\n")
        self.assertEqual(check(), 0)

    def test_progress_exposes_counts_without_private_tool_content(self):
        self.prepare()
        agent = self.packet / "run/agent"
        (agent / "actions").mkdir(parents=True)
        (agent / "requests").mkdir()
        demo.write_json(agent / "actions/0001.json", {"tool": "write_file", "result": {
            "returncode": 0, "output": "private source"}})
        demo.write_json(agent / "actions/0002.json", {"tool": "bash", "verification_candidate": True,
            "result": {"returncode": 1, "output": "private failure"}})
        (agent / "actions/0003.json").write_text('{"status":', encoding="utf-8")
        demo.write_json(agent / "requests/0001.json", {"prompt_tokens": 100, "completion_tokens": 50})
        demo.write_json(agent / "working-state.json", {"todos": [{"status": "done"}, {"status": "pending"}]})
        snapshot = demo.progress_snapshot(self.packet)
        self.assertEqual(snapshot, {"requests": 1, "tools": 3, "edits": 1, "checks": 1,
                                   "todos": 2, "todos_done": 1, "reported_tokens": 150})
        self.assertNotIn("private", str(snapshot))

    def test_relative_credential_path_resolves_against_package_not_cwd(self):
        (self.root / ".env").write_text("DEMO_TEST_KEY=sk-fake-relocated-only\n", encoding="utf-8")
        with patch.object(demo, "ROOT", self.root):
            environment = demo.credential_environment({"api_key_env": "DEMO_TEST_KEY", "dotenv_file": ".env"}, {})
        self.assertEqual(environment["DEMO_TEST_KEY"], "sk-fake-relocated-only")


if __name__ == "__main__":
    unittest.main()
