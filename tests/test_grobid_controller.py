"""Unit tests for GrobidController and GrobidAgent lifecycle management."""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from research_assistant.agents.grobid_controller import (
    GrobidAgent,
    GrobidController,
    GrobidStatus,
)


class TestGrobidController(unittest.TestCase):
    def setUp(self):
        self.controller = GrobidController(
            server_url="http://localhost:8070",
            docker_image="grobid/grobid:0.8.1",
            container_name="test-grobid",
        )

    def test_alias(self):
        self.assertIs(GrobidAgent, GrobidController)

    # ─── URL & Port Helpers ───────────────────────────────────────────────────

    def test_get_port(self):
        self.assertEqual(self.controller.get_port(), 8070)
        c2 = GrobidController(server_url="http://127.0.0.1:9090")
        self.assertEqual(c2.get_port(), 9090)

    def test_is_local_target(self):
        self.assertTrue(self.controller.is_local_target())
        c_ip = GrobidController(server_url="http://127.0.0.1:8070")
        self.assertTrue(c_ip.is_local_target())
        c_remote = GrobidController(server_url="https://grobid.hf.space")
        self.assertFalse(c_remote.is_local_target())

    def test_manual_command(self):
        cmd = self.controller.get_manual_command()
        self.assertIn("docker run", cmd)
        self.assertIn("8070:8070", cmd)
        self.assertIn("grobid/grobid:0.8.1", cmd)

    # ─── Health Probe (is_alive) ──────────────────────────────────────────────

    def test_is_alive_true(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "true"
        with patch("requests.get", return_value=mock_resp):
            self.assertTrue(self.controller.is_alive())

    def test_is_alive_false_when_text_not_true(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "false"
        with patch("requests.get", return_value=mock_resp):
            self.assertFalse(self.controller.is_alive())

    def test_is_alive_false_on_status_500(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "internal error"
        with patch("requests.get", return_value=mock_resp):
            self.assertFalse(self.controller.is_alive())

    def test_is_alive_false_on_connection_error(self):
        with patch("requests.get", side_effect=requests.ConnectionError("Refused")):
            self.assertFalse(self.controller.is_alive())

    # ─── Docker Environment Checks ────────────────────────────────────────────

    def test_check_docker_not_found(self):
        with patch("shutil.which", return_value=None):
            ok, msg = self.controller.check_docker()
            self.assertFalse(ok)
            self.assertIn("not found", msg.lower())

    def test_check_docker_success(self):
        mock_res = MagicMock(returncode=0, stdout="Client:\nServer:\n", stderr="")
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=mock_res):
            ok, msg = self.controller.check_docker()
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
            ok, msg = self.controller.check_docker()
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
            ok, msg = self.controller.check_docker()
            self.assertFalse(ok)
            self.assertIn("daemon is not running", msg.lower())

    def test_check_docker_timeout(self):
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=6)):
            ok, msg = self.controller.check_docker()
            self.assertFalse(ok)
            self.assertIn("timed out", msg.lower())

    # ─── Container Info ───────────────────────────────────────────────────────

    def test_get_container_info_running(self):
        mock_res = MagicMock(returncode=0, stdout="running\n", stderr="")
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            info = self.controller.get_container_info()
            self.assertTrue(info["exists"])
            self.assertEqual(info["status"], "running")

    def test_get_container_info_exited(self):
        mock_res = MagicMock(returncode=0, stdout="exited\n", stderr="")
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            info = self.controller.get_container_info()
            self.assertTrue(info["exists"])
            self.assertEqual(info["status"], "exited")

    def test_get_container_info_not_found(self):
        mock_res = MagicMock(returncode=1, stdout="", stderr="Error: No such object")
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            info = self.controller.get_container_info()
            self.assertFalse(info["exists"])
            self.assertEqual(info["status"], "not_found")

    def test_get_container_info_docker_down(self):
        with patch.object(self.controller, "check_docker", return_value=(False, "docker dead")):
            info = self.controller.get_container_info()
            self.assertFalse(info["exists"])
            self.assertEqual(info["status"], "unavailable")

    def test_get_available_grobid_images(self):
        mock_res = MagicMock(
            returncode=0,
            stdout="grobid/grobid:0.8.1\nlfoppiano/grobid:0.8.0\n<none>:<none>\npython:3.11\n",
            stderr="",
        )
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch("subprocess.run", return_value=mock_res):
            images = self.controller.get_available_grobid_images()
            self.assertEqual(images, ["grobid/grobid:0.8.1", "lfoppiano/grobid:0.8.0"])

    # ─── Overall Status Evaluation ────────────────────────────────────────────

    def test_check_status_running(self):
        with patch.object(self.controller, "is_alive", return_value=True), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "running"}):
            st = self.controller.check_status()
            self.assertTrue(st.is_alive)
            self.assertEqual(st.state, "RUNNING")
            self.assertTrue(st.docker_available)
            self.assertEqual(st.container_status, "running")

    def test_check_status_starting_container_up(self):
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "running"}):
            st = self.controller.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "STARTING")

    def test_check_status_stopped_container_exited(self):
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "exited"}):
            st = self.controller.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "STOPPED")

    def test_check_status_stopped_docker_down(self):
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(False, "daemon down")):
            st = self.controller.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "STOPPED")
            self.assertFalse(st.docker_available)
            self.assertIn("daemon down", st.details)

    def test_check_status_remote_server_unreachable(self):
        remote_c = GrobidController(server_url="https://remote.grobid.org")
        with patch.object(remote_c, "is_alive", return_value=False), \
             patch.object(remote_c, "check_docker", return_value=(True, "ok")):
            st = remote_c.check_status()
            self.assertFalse(st.is_alive)
            self.assertEqual(st.state, "ERROR")

    # ─── Starting Server ──────────────────────────────────────────────────────

    def test_start_server_already_running(self):
        with patch.object(self.controller, "is_alive", return_value=True):
            ok, msg = self.controller.start_server()
            self.assertTrue(ok)
            self.assertIn("already running", msg)

    def test_start_server_remote_host_fails(self):
        remote_c = GrobidController(server_url="https://grobid.example.com")
        with patch.object(remote_c, "is_alive", return_value=False):
            ok, msg = remote_c.start_server()
            self.assertFalse(ok)
            self.assertIn("remote host", msg.lower())

    def test_start_server_docker_not_available(self):
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(False, "Docker daemon not running")):
            ok, msg = self.controller.start_server()
            self.assertFalse(ok)
            self.assertIn("Docker daemon not running", msg)
            self.assertIn("docker run", msg)

    def test_start_server_starts_existing_stopped_container(self):
        start_res = MagicMock(returncode=0)
        with patch.object(self.controller, "is_alive", side_effect=[False, False, True]), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "exited"}), \
             patch("subprocess.run", return_value=start_res) as mock_sub:
            ok, msg = self.controller.start_server(timeout=5.0, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("successfully started", msg)
            # Verify docker start was called with container name
            mock_sub.assert_called()
            self.assertEqual(mock_sub.call_args[0][0], ["docker", "start", "test-grobid"])

    def test_start_server_runs_new_container_success(self):
        run_res = MagicMock(returncode=0)
        with patch.object(self.controller, "is_alive", side_effect=[False, True]), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": False, "status": "not_found"}), \
             patch.object(self.controller, "get_available_grobid_images", return_value=["grobid/grobid:0.8.1"]), \
             patch("subprocess.run", return_value=run_res) as mock_sub:
            ok, msg = self.controller.start_server(timeout=5.0, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("successfully started", msg)
            # Verify docker run was called
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0:2], ["docker", "run"])
            self.assertIn("test-grobid", args)
            self.assertIn("grobid/grobid:0.8.1", args)

    def test_start_server_handles_image_missing_error(self):
        run_res = MagicMock(returncode=1, stdout="", stderr="Unable to find image 'grobid/grobid:0.8.1' locally")
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": False, "status": "not_found"}), \
             patch.object(self.controller, "get_available_grobid_images", return_value=[]), \
             patch("subprocess.run", return_value=run_res):
            ok, msg = self.controller.start_server(timeout=5.0, poll_interval=0.01)
            self.assertFalse(ok)
            self.assertIn("docker pull", msg)

    def test_start_server_handles_port_conflict(self):
        run_res = MagicMock(returncode=1, stdout="", stderr="Bind for 0.0.0.0:8070 failed: port is already allocated")
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": False, "status": "not_found"}), \
             patch.object(self.controller, "get_available_grobid_images", return_value=[]), \
             patch("subprocess.run", return_value=run_res):
            ok, msg = self.controller.start_server(timeout=5.0, poll_interval=0.01)
            self.assertFalse(ok)
            self.assertIn("Port 8070 is already in use", msg)

    def test_start_server_container_runs_but_jvm_initializing(self):
        run_res = MagicMock(returncode=0)
        # is_alive always returns False, but container status is running
        with patch.object(self.controller, "is_alive", return_value=False), \
             patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "running"}), \
             patch("subprocess.run", return_value=run_res):
            ok, msg = self.controller.start_server(timeout=0.05, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("initializing", msg.lower())

    # ─── Stopping Server ──────────────────────────────────────────────────────

    def test_stop_server_running_container(self):
        stop_res = MagicMock(returncode=0)
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "running"}), \
             patch("subprocess.run", return_value=stop_res) as mock_sub:
            ok, msg = self.controller.stop_server()
            self.assertTrue(ok)
            self.assertIn("stopped successfully", msg)
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0:2], ["docker", "stop"])
            self.assertIn("test-grobid", args)

    def test_stop_server_already_stopped(self):
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": True, "status": "exited"}):
            ok, msg = self.controller.stop_server()
            self.assertTrue(ok)
            self.assertIn("already stopped", msg)

    def test_stop_server_running_outside_container_warns(self):
        with patch.object(self.controller, "check_docker", return_value=(True, "ok")), \
             patch.object(self.controller, "get_container_info", return_value={"exists": False, "status": "not_found"}), \
             patch.object(self.controller, "is_alive", return_value=True):
            ok, msg = self.controller.stop_server()
            self.assertFalse(ok)
            self.assertIn("external service", msg.lower())

    def test_stop_server_docker_unavailable_when_stopped(self):
        with patch.object(self.controller, "check_docker", return_value=(False, "no docker")), \
             patch.object(self.controller, "is_alive", return_value=False):
            ok, msg = self.controller.stop_server()
            self.assertTrue(ok)
            self.assertIn("already stopped", msg)

    # ─── Custom Process Start / Stop ──────────────────────────────────────────

    def test_custom_start_and_stop_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = os.path.join(tmpdir, "grobid_process.pid")
            controller = GrobidController(
                server_url="http://localhost:8070",
                start_command="echo 'starting'",
            )
            controller._pid_file = pid_file

            mock_proc = MagicMock(pid=12345)
            with patch.object(controller, "is_alive", side_effect=[False, True]), \
                 patch("subprocess.Popen", return_value=mock_proc):
                ok, msg = controller.start_server(timeout=2.0, poll_interval=0.01)
                self.assertTrue(ok)
                self.assertTrue(os.path.exists(pid_file))
                with open(pid_file) as f:
                    self.assertEqual(f.read(), "12345")

            with patch("os.kill") as mock_kill:
                ok, msg = controller.stop_server()
                self.assertTrue(ok)
                mock_kill.assert_called_once()
                self.assertFalse(os.path.exists(pid_file))

    # ─── App UI Integration ───────────────────────────────────────────────────

    def test_render_grobid_controls_in_app(self):
        import app
        # Call _render_grobid_controls directly to ensure it does not raise
        with patch.object(self.controller, "check_status", return_value=GrobidStatus(
            is_alive=False,
            state="STOPPED",
            message="stopped",
            server_url="http://localhost:8070",
            docker_available=True,
            container_name="grobid",
        )):
            try:
                app._render_grobid_controls()
            except Exception as e:
                self.fail(f"_render_grobid_controls crashed with: {e}")


if __name__ == "__main__":
    unittest.main()
