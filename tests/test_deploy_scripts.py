"""deploy/local.sh.

The script is run for real, under bash, with `docker`, `curl` and the Python
interpreter replaced by stubs that record what they were asked to do. Nothing is
started, pulled or removed. What the tests pin: local.sh binds the app to 127.0.0.1,
reuses the container name the app's own GROBID buttons use, and never touches any
other container.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"

# Each stub appends its argv to $CALLS. Behaviour is switched by files in $STATE,
# so a test says "Docker is down" by touching docker_down, not by patching code.
STUBS = {
    "curl": r'''
        echo "curl $*" >> "$CALLS"
        case "$*" in
          *"/api/isalive"*) [ -e "$STATE/grobid_alive" ] ;;
          *"/api/tags"*)    [ -e "$STATE/ollama" ] && echo '{"models":[{"name":"a"},{"name":"b"}]}' ;;
          *)                exit 0 ;;
        esac
    ''',
    "sleep": "exit 0",
    "docker": r'''
        echo "docker $*" >> "$CALLS"
        case "$1" in
          info)    [ ! -e "$STATE/docker_down" ]; exit $? ;;
          inspect) f="$STATE/container_${!#}"; [ -e "$f" ] || exit 1; cat "$f"; exit 0 ;;
          run|start)
            if [ "$1" = run ]; then name=""; prev=""
              for a in "$@"; do [ "$prev" = "--name" ] && name="$a"; prev="$a"; done
            else name="$2"; fi
            if [ -e "$STATE/dies_on_start" ]; then echo exited > "$STATE/container_$name"
            else echo running > "$STATE/container_$name"
                 if [ -e "$STATE/start_works" ]; then touch "$STATE/grobid_alive"; fi; fi ;;
          stop)    echo exited > "$STATE/container_$2"; rm -f "$STATE/grobid_alive" ;;
          logs)    echo "stub log line" ;;
          ps)      cat "$STATE/containers" 2>/dev/null || true ;;
        esac
        exit 0
    ''',
    "python": r'''
        echo "python $*" >> "$CALLS"
        [ "$1" = "-c" ] && [ -e "$STATE/no_streamlit" ] && exit 1
        exit 0
    ''',
}


@unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "needs bash")
class ScriptHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin, self.state, self.calls = self.tmp / "bin", self.tmp / "state", self.tmp / "calls.log"
        self.bin.mkdir()
        self.state.mkdir()
        self.calls.touch()
        for name, body in STUBS.items():
            path = self.bin / name
            path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body).strip() + "\n")
            path.chmod(0o755)

    def flag(self, name, content=""):
        (self.state / name).write_text(content)

    def run_script(self, script, *args, **env):
        full = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.tmp),
            "CALLS": str(self.calls),
            "STATE": str(self.state),
            "PYTHON": str(self.bin / "python"),
            "GROBID_WAIT": "3",
            **env,
        }
        return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=full, timeout=60)

    def log(self):
        return self.calls.read_text().splitlines()

    def calls_starting(self, prefix):
        return [line for line in self.log() if line.startswith(prefix)]


class TestLocalScript(ScriptHarness):
    def setUp(self):
        super().setUp()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = str(probe.getsockname()[1])

    def up(self, *args, **env):
        return self.run_script(DEPLOY / "local.sh", "up", *args, PORT=self.port, **env)

    def app_launches(self):
        return self.calls_starting("python -m streamlit run")

    def test_the_app_listens_on_loopback_only(self):
        self.flag("grobid_alive")
        result = self.up()
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        (launch,) = self.app_launches()
        self.assertIn("app.py", launch)
        self.assertIn("--server.address 127.0.0.1", launch)
        self.assertIn(f"--server.port {self.port}", launch)
        self.assertNotIn("0.0.0.0", launch)

    def test_a_running_grobid_is_left_alone(self):
        self.flag("grobid_alive")
        self.up()
        self.assertEqual(self.calls_starting("docker"), [])

    def test_a_missing_container_is_created_on_the_apps_name_bound_to_loopback(self):
        self.flag("start_works")
        result = self.up()
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        (run,) = self.calls_starting("docker run")
        self.assertIn("--name grobid", run)
        self.assertIn("-p 127.0.0.1:8070:8070", run)
        self.assertTrue(run.endswith("grobid/grobid:0.8.1"), run)
        self.assertEqual(len(self.app_launches()), 1)

    def test_an_existing_stopped_container_is_started_not_recreated(self):
        # This Mac has exactly this: a stopped `grobid` container built from a different image.
        self.flag("container_grobid", "exited\n")
        self.flag("start_works")
        self.up()
        self.assertEqual(self.calls_starting("docker start"), ["docker start grobid"])
        self.assertEqual(self.calls_starting("docker run"), [])

    def test_container_name_and_port_follow_the_apps_own_settings(self):
        self.flag("start_works")
        self.up(GROBID_CONTAINER_NAME="my-grobid", GROBID_SERVER="http://localhost:9070", GROBID_DOCKER_IMAGE="img:1")
        (run,) = self.calls_starting("docker run")
        self.assertIn("--name my-grobid", run)
        self.assertIn("-p 127.0.0.1:9070:8070", run)
        self.assertTrue(run.endswith("img:1"), run)

    def test_without_docker_the_app_still_starts_and_says_what_is_lost(self):
        self.flag("docker_down")
        result = self.up()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(self.app_launches()), 1)
        self.assertIn("Docker is not running", result.stderr)
        self.assertEqual([c for c in self.calls_starting("docker") if not c.startswith("docker info")], [])

    def test_a_container_that_dies_on_start_is_reported_and_the_app_still_starts(self):
        self.flag("dies_on_start")
        result = self.up()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(self.app_launches()), 1)
        self.assertIn("stopped right after starting", result.stderr)
        self.assertIn("137", result.stderr)  # the OOM kill this Mac has already produced once
        self.assertIn("stub log line", result.stderr)

    def test_a_grobid_that_never_answers_does_not_block_the_app(self):
        result = self.up(GROBID_WAIT="2")  # container "starts" but start_works is off, so nothing ever answers
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(self.app_launches()), 1)
        self.assertIn("did not answer within 2s", result.stderr)

    def test_no_grobid_flag_makes_no_docker_calls(self):
        result = self.up("--no-grobid")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls_starting("docker"), [])
        self.assertEqual(len(self.app_launches()), 1)

    def test_a_remote_grobid_is_never_managed(self):
        result = self.up(GROBID_SERVER="https://grobid.example.org")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls_starting("docker"), [])
        self.assertIn("not on this machine", result.stderr)

    def test_a_taken_port_stops_before_anything_is_launched(self):
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            result = self.run_script(DEPLOY / "local.sh", "up", PORT=str(held.getsockname()[1]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already taken", result.stderr)
        self.assertEqual(self.app_launches(), [])
        self.assertEqual(self.calls_starting("docker"), [])

    def test_an_interpreter_without_streamlit_gets_a_clear_message(self):
        self.flag("no_streamlit")
        result = self.up()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PYTHON=", result.stderr)
        self.assertEqual(self.app_launches(), [])

    def test_an_unknown_option_is_refused(self):
        result = self.up("--everything")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.app_launches(), [])

    def test_down_stops_only_the_grobid_container(self):
        self.flag("container_grobid", "running\n")
        self.flag("container_postgres", "running\n")
        result = self.run_script(DEPLOY / "local.sh", "down")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls_starting("docker stop"), ["docker stop grobid"])
        self.assertEqual(self.calls_starting("docker rm"), [])
        self.assertEqual((self.state / "container_postgres").read_text().strip(), "running")

    def test_down_with_nothing_running_does_nothing(self):
        result = self.run_script(DEPLOY / "local.sh", "down")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls_starting("docker stop"), [])

    def test_status_reports_all_three_and_changes_nothing(self):
        self.flag("grobid_alive")
        self.flag("ollama")
        self.flag("container_grobid", "running\n")
        result = self.run_script(DEPLOY / "local.sh", "status", PORT=self.port)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        self.assertIn(f"http://127.0.0.1:{self.port}", out)
        self.assertRegex(out, r"grobid .*alive \(container grobid: running\)")
        self.assertRegex(out, r"ollama .*up \(2 models installed\)")
        changing = [c for c in self.calls_starting("docker") if c.split()[1] in ("run", "start", "stop", "rm")]
        self.assertEqual(changing, [])
        self.assertEqual(self.app_launches(), [])

    def test_no_arguments_prints_usage_and_fails(self):
        result = self.run_script(DEPLOY / "local.sh")
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage", result.stderr)


if __name__ == "__main__":
    unittest.main()
