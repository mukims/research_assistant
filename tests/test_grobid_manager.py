"""Unit tests for GrobidManager and GrobidController lifecycle management."""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from research_assistant.shared.grobid_manager import (
    GrobidAgent,
    GrobidController,
    GrobidManager,
    GrobidStatus,
    check_grobid_status,
    is_grobid_alive,
    restart_grobid,
    start_grobid,
    stop_grobid,
)


class TestGrobidManager(unittest.TestCase):
    def setUp(self):
        self.manager = GrobidManager(
            server_url="http://localhost:8070",
            docker_image="grobid/grobid:0.8.1",
            container_name="test-grobid",
        )

    def test_aliases(self):
        self.assertIs(GrobidAgent, GrobidManager)
        self.assertIs(GrobidController, GrobidManager)

    # ─── URL & Port Helpers ───────────────────────────────────────────────────

    def test_get_port(self):
        self.assertEqual(self.manager.get_port(), 8070)
        m2 = GrobidManager(server_url="http://127.0.0.1:9090")
        self.assertEqual(m2.get_port(), 9090)

    def test_is_local_target(self):
        self.assertTrue(self.manager.is_local_target())
        m_ip = GrobidManager(server_url="http://127.0.0.1:8070")
        self.assertTrue(m_ip.is_local_target())
        m_remote = GrobidManager(server_url="https://grobid.hf.space")
        self.assertFalse(m_remote.is_local_target())

    def test_manual_command(self):
        cmd = self.manager.get_manual_command()
        self.assertIn("docker run", cmd)
        self.assertIn("8070:8070", cmd)
        self.assertIn("grobid/grobid:0.8.1", cmd)

    # ─── Health Probing ───────────────────────────────────────────────────────

    def test_is_alive_true(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "true"
        with patch("requests.get", return_value=mock_resp):
            self.assertTrue(self.manager.is_alive())

    def test_is_alive_false_when_text_not_true(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "false"
        with patch("requests.get", return_value=mock_resp):
            self.assertFalse(self.manager.is_alive())

    def test_is_alive_false_on_status_500(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "internal error"
        with patch("requests.get", return_value=mock_resp):
            self.assertFalse(self.manager.is_alive())

    def test_is_alive_false_on_connection_error(self):
        with patch("requests.get", side_effect=requests.ConnectionError("Refused")):
            self.assertFalse(self.manager.is_alive())

    def test_is_alive_with_extractor_probe(self):
        with patch("research_assistant.agents.agent1_extractor.grobid_alive", return_value=True) as mock_ga:
            alive = self.manager.is_alive(use_extractor_probe=True)
            self.assertTrue(alive)
            mock_ga.assert_called_once()

    def test_check_alive_via_extractor(self):
        with patch("research_assistant.agents.agent1_extractor.grobid_alive", return_value=True) as mock_ga:
            alive = self.manager.check_alive_via_extractor()
            self.assertTrue(alive)
            mock_ga.assert_called_once()

    # ─── Docker Checks ────────────────────────────────────────────────────────

    def test_check_docker_not_found(self):
        with patch("shutil.which", return_value=None):
            ok, msg = self.manager.check_docker()
            self.assertFalse(ok)
            self.assertIn("not found", msg.lower())

    def test_check_docker_success(self):
        mock_res = MagicMock(returncode=0, stdout="Client:\nServer:\n", stderr="")
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=mock_res):
            ok, msg = self.manager.check_docker()
            self.assertTrue(ok)
            self.assertIn("running", msg.lower())

    def test_check_docker_permission_denied(self):
        mock_res = MagicMock(
            returncode=1,
            stdout="",
            stderr="permission denied while trying to connect to the docker API at unix:///var/run/docker.sock",
        )
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=mock_res):
            ok, msg = self.manager.check_docker()
            self.assertFalse(ok)
            self.assertIn("permission denied", msg.lower())

    def test_check_docker_daemon_not_running(self):
        mock_res = MagicMock(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
        )
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=mock_res):
            ok, msg = self.manager.check_docker()
            self.assertFalse(ok)
            self.assertIn("daemon is not running", msg.lower())

    def test_check_docker_timeout(self):
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=6)):
            ok, msg = self.manager.check_docker()
            self.assertFalse(ok)
            self.assertIn("timed out", msg.lower())

    # ─── Container Discovery & Info ───────────────────────────────────────────

    def test_get_container_info_named_container_running(self):
        mock_res = MagicMock(returncode=0, stdout="running\n", stderr="")
        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            info = self.manager.get_container_info()
            self.assertTrue(info["exists"])
            self.assertEqual(info["status"], "running")
            self.assertEqual(info["name"], "test-grobid")

    def test_get_container_info_discovers_active_container(self):
        # inspect for test-grobid fails, but ps -a finds a container on port 8070
        inspect_fail = MagicMock(returncode=1, stdout="", stderr="no such object")
        ps_out = "abc123\tlaughing_williams\trunning\tlfoppiano/grobid:0.8.0\t0.0.0.0:8070->8070/tcp\n"
        ps_res = MagicMock(returncode=0, stdout=ps_out, stderr="")

        def sub_run(cmd, *args, **kwargs):
            if cmd[1] == "inspect":
                return inspect_fail
            if cmd[1] == "ps":
                return ps_res
            return MagicMock(returncode=0)

        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", side_effect=sub_run):
            info = self.manager.get_container_info()
            self.assertTrue(info["exists"])
            self.assertEqual(info["status"], "running")
            self.assertEqual(info["name"], "laughing_williams")

    def test_get_available_grobid_images(self):
        mock_res = MagicMock(
            returncode=0,
            stdout="grobid/grobid:0.8.1\nlfoppiano/grobid:0.8.0\n<none>:<none>\npython:3.11\n",
            stderr="",
        )
        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            images = self.manager.get_available_grobid_images()
            self.assertEqual(images, ["grobid/grobid:0.8.1", "lfoppiano/grobid:0.8.0"])

    # ─── Overall Status Evaluation ────────────────────────────────────────────

    def test_check_status_running(self):
        with patch.object(self.manager, "is_alive", return_value=True), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "running", "name": "test-grobid"}):
            st = self.manager.check_status()
            self.assertTrue(st.is_alive)
            self.assertEqual(st.state, "RUNNING")
            self.assertTrue(st.docker_available)
            self.assertEqual(st.container_status, "running")

    def test_check_status_starting_container_up(self):
        with patch.object(self.manager, "is_alive", return_value=False), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "running", "name": "test-grobid"}):
            st = self.manager.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "STARTING")

    def test_check_status_stopped_container_exited(self):
        with patch.object(self.manager, "is_alive", return_value=False), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "exited", "name": "test-grobid"}):
            st = self.manager.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "STOPPED")

    def test_check_status_remote_server_unreachable(self):
        remote_m = GrobidManager(server_url="https://remote.grobid.org")
        with patch.object(remote_m, "is_alive", return_value=False), \
             patch.object(remote_m, "check_docker", return_value=(True, "ok")):
            st = remote_m.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "ERROR")

    # ─── Starting Server ──────────────────────────────────────────────────────

    def test_start_server_already_running(self):
        with patch.object(self.manager, "is_alive", return_value=True):
            ok, msg = self.manager.start_server()
            self.assertTrue(ok)
            self.assertIn("already running", msg)

    def test_start_server_remote_host_fails(self):
        remote_m = GrobidManager(server_url="https://grobid.example.com")
        with patch.object(remote_m, "is_alive", return_value=False):
            ok, msg = remote_m.start_server()
            self.assertFalse(ok)
            self.assertIn("remote host", msg.lower())

    def test_start_server_docker_not_available(self):
        with patch.object(self.manager, "is_alive", return_value=False), \
             patch.object(self.manager, "check_docker", return_value=(False, "Docker daemon not running")):
            ok, msg = self.manager.start_server()
            self.assertFalse(ok)
            self.assertIn("Docker daemon not running", msg)

    def test_start_server_starts_existing_stopped_container(self):
        start_res = MagicMock(returncode=0)
        with patch.object(self.manager, "is_alive", side_effect=[False, False, True]), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "exited", "name": "test-grobid"}), \
             patch("subprocess.run", return_value=start_res) as mock_sub:
            ok, msg = self.manager.start_server(timeout=5.0, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("successfully started", msg)
            mock_sub.assert_called()
            self.assertEqual(mock_sub.call_args[0][0], ["docker", "start", "test-grobid"])

    def test_start_server_runs_new_container_success(self):
        run_res = MagicMock(returncode=0)
        with patch.object(self.manager, "is_alive", side_effect=[False, True]), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": False, "status": "not_found", "name": "test-grobid"}), \
             patch.object(self.manager, "get_available_grobid_images", return_value=["grobid/grobid:0.8.1"]), \
             patch("subprocess.run", return_value=run_res) as mock_sub:
            ok, msg = self.manager.start_server(timeout=5.0, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("successfully started", msg)
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0:2], ["docker", "run"])
            self.assertIn("test-grobid", args)

    def test_start_server_uses_cached_image_if_default_missing(self):
        run_res = MagicMock(returncode=0)
        with patch.object(self.manager, "is_alive", side_effect=[False, True]), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": False, "status": "not_found", "name": "test-grobid"}), \
             patch.object(self.manager, "get_available_grobid_images", return_value=["lfoppiano/grobid:0.8.1"]), \
             patch("subprocess.run", return_value=run_res) as mock_sub:
            ok, msg = self.manager.start_server(timeout=5.0, poll_interval=0.01)
            self.assertTrue(ok)
            args = mock_sub.call_args[0][0]
            self.assertIn("lfoppiano/grobid:0.8.1", args)

    def test_start_server_container_running_jvm_initializing(self):
        run_res = MagicMock(returncode=0)
        with patch.object(self.manager, "is_alive", return_value=False), \
             patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "running", "name": "test-grobid"}), \
             patch("subprocess.run", return_value=run_res):
            ok, msg = self.manager.start_server(timeout=0.05, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("initializing", msg.lower())

    # ─── Stopping Server ──────────────────────────────────────────────────────

    def test_stop_server_running_container(self):
        stop_res = MagicMock(returncode=0)
        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "running", "name": "test-grobid"}), \
             patch("subprocess.run", return_value=stop_res) as mock_sub:
            ok, msg = self.manager.stop_server()
            self.assertTrue(ok)
            self.assertIn("stopped successfully", msg)
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0:2], ["docker", "stop"])
            self.assertIn("test-grobid", args)

    def test_stop_server_stops_discovered_container(self):
        stop_res = MagicMock(returncode=0)
        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "running", "name": "laughing_williams"}), \
             patch("subprocess.run", return_value=stop_res) as mock_sub:
            ok, msg = self.manager.stop_server()
            self.assertTrue(ok)
            self.assertIn("stopped successfully", msg)
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0:2], ["docker", "stop"])
            self.assertIn("laughing_williams", args)

    def test_stop_server_already_stopped(self):
        with patch.object(self.manager, "check_docker", return_value=(True, "ok")), \
             patch.object(self.manager, "get_container_info", return_value={"exists": True, "status": "exited", "name": "test-grobid"}):
            ok, msg = self.manager.stop_server()
            self.assertTrue(ok)
            self.assertIn("already stopped", msg)

    # ─── Custom Process & JAR Startup ─────────────────────────────────────────

    def test_custom_jar_path_configuration(self):
        manager = GrobidManager(
            server_url="http://localhost:8070",
            jar_path="/opt/grobid/grobid-service.jar",
        )
        self.assertIn("java -Xmx4g -jar /opt/grobid/grobid-service.jar", manager.start_command)

    def test_custom_start_and_stop_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = os.path.join(tmpdir, "grobid_process.pid")
            manager = GrobidManager(
                server_url="http://localhost:8070",
                start_command="echo 'launching'",
            )
            manager._pid_file = pid_file

            mock_proc = MagicMock(pid=54321)
            with patch.object(manager, "is_alive", side_effect=[False, True]), \
                 patch("subprocess.Popen", return_value=mock_proc):
                ok, msg = manager.start_server(timeout=2.0, poll_interval=0.01)
                self.assertTrue(ok)
                self.assertTrue(os.path.exists(pid_file))
                with open(pid_file) as f:
                    self.assertEqual(f.read(), "54321")

            with patch("os.kill") as mock_kill:
                ok, msg = manager.stop_server()
                self.assertTrue(ok)
                mock_kill.assert_called_once()
                self.assertFalse(os.path.exists(pid_file))

    # ─── Module Functions ─────────────────────────────────────────────────────

    def test_module_convenience_functions(self):
        with patch.object(GrobidManager, "is_alive", return_value=True):
            self.assertTrue(is_grobid_alive())

        with patch.object(GrobidManager, "check_status", return_value=GrobidStatus(
            is_alive=True,
            state="RUNNING",
            message="ok",
            server_url="http://localhost:8070",
            docker_available=True,
            container_name="grobid",
        )):
            st = check_grobid_status()
            self.assertEqual(st.state, "RUNNING")

        with patch.object(GrobidManager, "start_server", return_value=(True, "started")):
            ok, msg = start_grobid()
            self.assertTrue(ok)

        with patch.object(GrobidManager, "stop_server", return_value=(True, "stopped")):
            ok, msg = stop_grobid()
            self.assertTrue(ok)

        with patch.object(GrobidManager, "restart_server", return_value=(True, "restarted")):
            ok, msg = restart_grobid()
            self.assertTrue(ok)

    # ─── Streamlit UI Integration ─────────────────────────────────────────────

    def test_ui_render_running_state(self):
        import streamlit as st
        import app
        st.session_state["grobid_feedback"] = ("success", "GROBID started successfully!")
        with patch.object(GrobidManager, "check_status", return_value=GrobidStatus(
            is_alive=True,
            state="RUNNING",
            message="Server running",
            server_url="http://localhost:8070",
            docker_available=True,
            container_name="test-grobid",
        )):
            try:
                app._render_grobid_controls()
                self.assertEqual(st.session_state.get("grobid_server_state"), "RUNNING")
            except Exception as exc:
                self.fail(f"UI render crashed in running state: {exc}")

    def test_ui_render_stopped_state(self):
        import streamlit as st
        import app
        with patch.object(GrobidManager, "check_status", return_value=GrobidStatus(
            is_alive=False,
            state="STOPPED",
            message="Server stopped",
            server_url="http://localhost:8070",
            docker_available=True,
            container_name="test-grobid",
        )):
            try:
                app._render_grobid_controls()
                self.assertEqual(st.session_state.get("grobid_server_state"), "STOPPED")
            except Exception as exc:
                self.fail(f"UI render crashed in stopped state: {exc}")

    def test_ui_render_starting_state(self):
        import streamlit as st
        import app
        with patch.object(GrobidManager, "check_status", return_value=GrobidStatus(
            is_alive=False,
            state="STARTING",
            message="Starting JVM",
            server_url="http://localhost:8070",
            docker_available=True,
            container_name="test-grobid",
        )):
            try:
                app._render_grobid_controls()
                self.assertEqual(st.session_state.get("grobid_server_state"), "STARTING")
            except Exception as exc:
                self.fail(f"UI render crashed in starting state: {exc}")

    def test_grobid_ok_cache_function(self):
        import app
        with patch.object(GrobidManager, "is_alive", return_value=True):
            app._clear_grobid_cache()
            self.assertTrue(app._grobid_ok())


if __name__ == "__main__":
    unittest.main()
