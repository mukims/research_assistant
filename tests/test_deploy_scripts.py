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
          *"/api/v2/heartbeat"*) [ ! -e "$STATE/chroma_down" ] ;;
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
        case "$*" in *" run --path "*) [ -e "$STATE/chroma_starts" ] && rm -f "$STATE/chroma_down"; exit 0 ;; esac
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
            "CITATION_DATA_DIR": str(self.tmp / "data"),  # never the real data/: up writes chroma.pid there
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



    # ── The Chroma server: the one process that opens the vector store ──

    def chroma_launches(self):
        return [line for line in self.log() if " run --path " in line]

    def test_a_stopped_chroma_server_is_started_on_loopback_before_the_app(self):
        self.flag("grobid_alive")
        self.flag("chroma_down")
        self.flag("chroma_starts")
        result = self.up()
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        (launch,) = self.chroma_launches()
        self.assertIn(f"run --path {self.tmp / 'data' / 'physics_vectordb'} --host 127.0.0.1 --port 8000", launch)
        self.assertTrue((self.tmp / "data" / "chroma.pid").exists())
        self.assertLess(self.log().index(launch), self.log().index(self.app_launches()[0]))

    def test_a_running_chroma_server_is_left_alone(self):
        self.flag("grobid_alive")
        self.up()
        self.assertEqual(self.chroma_launches(), [])

    def test_a_chroma_server_that_never_answers_stops_before_the_app(self):
        self.flag("grobid_alive")
        self.flag("chroma_down")
        result = self.up(CHROMA_WAIT="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not answer within 2s", result.stderr)
        self.assertEqual(self.app_launches(), [])

    def test_a_remote_chroma_server_is_never_started(self):
        self.flag("chroma_down")
        result = self.up(CITATION_CHROMA_SERVER="https://chroma.example.org")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not on this machine", result.stderr)
        self.assertEqual(self.chroma_launches(), [])

    def test_the_chroma_command_starts_only_the_server(self):
        self.flag("chroma_down")
        self.flag("chroma_starts")
        result = self.run_script(DEPLOY / "local.sh", "chroma")
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        self.assertEqual(len(self.chroma_launches()), 1)
        self.assertEqual(self.app_launches(), [])
        self.assertEqual(self.calls_starting("docker"), [])

    def test_down_stops_the_server_it_started_and_nothing_else(self):
        server = subprocess.Popen(["/bin/sleep", "30"])
        self.addCleanup(server.kill)
        (self.tmp / "data").mkdir()
        (self.tmp / "data" / "chroma.pid").write_text(f"{server.pid}\n")
        result = self.run_script(DEPLOY / "local.sh", "down")
        self.assertEqual(server.wait(timeout=5), -15)
        self.assertFalse((self.tmp / "data" / "chroma.pid").exists())
        self.assertIn("Stopping the Chroma server", result.stdout)

    def test_down_without_a_pid_file_stops_nothing(self):
        result = self.run_script(DEPLOY / "local.sh", "down")
        self.assertIn("nothing to stop", result.stdout)

    def test_status_reports_the_chroma_server(self):
        result = self.run_script(DEPLOY / "local.sh", "status")
        self.assertIn("chroma  http://localhost:8000   alive", result.stdout)


class TestMigrateSetupScript(ScriptHarness):
    """deploy/migrate_setup.sh, copied into a throwaway project root so the real .env is never read."""

    def setUp(self):
        super().setUp()
        self.root = self.tmp / "project"
        (self.root / "deploy").mkdir(parents=True)
        shutil.copy(DEPLOY / "migrate_setup.sh", self.root / "deploy" / "migrate_setup.sh")
        (self.root / "requirements.txt").write_text("# none\n")
        self.venv = self.tmp / "venv"
        stubs = {
            # The interpreter: `-m venv DIR` makes DIR/bin/{python,pip} stubs; version checks pass unless py313 is set.
            "python": r'''
                echo "python $*" >> "$CALLS"
                if [ "$1" = "-V" ]; then echo "Python 3.12.1"; exit 0; fi
                if [ "$1" = "-c" ]; then [ -e "$STATE/py313" ] && exit 1; exit 0; fi
                if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
                  mkdir -p "$3/bin"
                  printf '#!/usr/bin/env bash\necho "venvpython $*" >> "$CALLS"\n[ "$1" = "-c" ] && [ ! -e "$STATE/packages" ] && exit 1\nexit 0\n' > "$3/bin/python"
                  printf '#!/usr/bin/env bash\necho "pip $*" >> "$CALLS"\ntouch "$STATE/packages"\nexit 0\n' > "$3/bin/pip"
                  chmod +x "$3/bin/python" "$3/bin/pip"
                fi
                exit 0
            ''',
            "ollama": r'''
                echo "ollama $*" >> "$CALLS"
                [ "$1" = "pull" ] && touch "$STATE/model"
                exit 0
            ''',
            "curl": r'''
                echo "curl $*" >> "$CALLS"
                case "$*" in
                  *"/api/isalive"*) [ -e "$STATE/grobid_alive" ] ;;
                  *"/api/version"*) [ -e "$STATE/ollama" ] && printf '{"version":"%s"}' "$(cat "$STATE/ollama")" ;;
                  *"/api/tags"*)    [ -e "$STATE/ollama" ] || exit 22
                                    if [ -e "$STATE/model" ]; then echo '{"models":[{"name":"embeddinggemma-2:latest","digest":"eaff2253e2ba0000"}]}'; else echo '{"models":[]}'; fi ;;
                  *)                exit 0 ;;
                esac
            ''',
        }
        for name, body in stubs.items():
            path = self.bin / name
            path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body).strip() + "\n")
            path.chmod(0o755)
        self.flag("ollama", "0.40.5")

    def setup(self, *args, **env):
        return self.run_script(self.root / "deploy" / "migrate_setup.sh", "--venv", str(self.venv), *args,
                               CITATION_DATA_DIR=str(self.tmp / "data"), **env)

    def test_a_bare_machine_gets_a_venv_the_model_and_a_loopback_only_grobid(self):
        self.flag("start_works")
        result = self.setup()
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        self.assertEqual(len(self.calls_starting("pip install -q -r")), 1)
        self.assertEqual(self.calls_starting("ollama pull"), ["ollama pull embeddinggemma-2"])
        (run,) = self.calls_starting("docker run")
        self.assertIn("-p 127.0.0.1:8070:8070", run)
        self.assertIn("--name grobid", run)
        self.assertTrue(run.endswith("grobid/grobid:0.8.1"), run)
        self.assertIn("ready. Next:", result.stdout)

    def test_a_prepared_machine_is_left_alone(self):
        self.flag("model")
        self.flag("grobid_alive")
        self.flag("packages")
        (self.venv / "bin").mkdir(parents=True)
        for n in ("python", "pip"):
            (self.venv / "bin" / n).write_text(f"#!/usr/bin/env bash\necho \"{n} $*\" >> \"$CALLS\"\nexit 0\n")
            (self.venv / "bin" / n).chmod(0o755)
        result = self.setup()
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        self.assertEqual(self.calls_starting("ollama pull"), [])
        self.assertEqual(self.calls_starting("docker"), [])
        self.assertEqual(self.calls_starting("pip"), [])

    def test_check_only_changes_nothing(self):
        result = self.setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        self.assertFalse(self.venv.exists())
        self.assertEqual(self.calls_starting("ollama pull"), [])
        self.assertEqual(self.calls_starting("docker run"), [])
        self.assertEqual(self.calls_starting("docker start"), [])
        self.assertIn("would:", result.stdout)

    def test_an_ollama_older_than_the_model_needs_is_refused(self):
        self.flag("ollama", "0.35.9")
        result = self.setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("older than 0.36.0", result.stderr)
        self.assertEqual(self.calls_starting("ollama pull"), [])

    def test_ollama_that_is_not_running_gets_a_clear_message(self):
        (self.state / "ollama").unlink()
        result = self.setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not answering", result.stderr)

    def test_python_313_is_refused(self):
        self.flag("py313")
        result = self.setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("3.10-3.12", result.stderr)

    def test_a_pinned_embedder_in_the_data_dir_env_is_warned_about(self):
        (self.tmp / "data").mkdir()
        (self.tmp / "data" / ".env").write_text("CITATION_EMBED_MODEL=nomic-embed-text\n")
        self.flag("model"); self.flag("grobid_alive"); self.flag("packages")
        result = self.setup("--skip-grobid")
        self.assertIn("overrides the embeddinggemma-2 default", result.stderr)

    def test_no_summary_model_is_warned_about(self):
        self.flag("model")
        result = self.setup("--skip-grobid")
        self.assertIn("no summary model", result.stderr)

    def test_an_unknown_option_is_refused(self):
        self.assertNotEqual(self.setup("--nope").returncode, 0)


if __name__ == "__main__":
    unittest.main()
