"""
GROBID server management module and controller agent.

Provides lifecycle management, status inspection, and health probing
for the GROBID server (Docker, local JAR, or external service).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from research_assistant import config
from research_assistant.shared.log import get_logger

logger = get_logger("grobid_manager")


@dataclass
class GrobidStatus:
    is_alive: bool
    state: str  # "RUNNING" | "STOPPED" | "STARTING" | "ERROR"
    message: str
    server_url: str
    docker_available: bool
    container_name: str
    container_status: Optional[str] = None
    image: Optional[str] = None
    details: Optional[str] = None
    manual_command: str = ""


class GrobidManager:
    """Manager and controller for the GROBID service lifecycle.

    Supports running via Docker (default container grobid/grobid:0.8.1),
    local JAR execution, or custom start command.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        container_name: Optional[str] = None,
        start_command: Optional[str] = None,
        jar_path: Optional[str] = None,
    ) -> None:
        self.server_url = (server_url or config.GROBID_SERVER).rstrip("/")
        self.docker_image = docker_image or getattr(config, "GROBID_DOCKER_IMAGE", "grobid/grobid:0.8.1")
        self.container_name = container_name or getattr(config, "GROBID_CONTAINER_NAME", "grobid")
        self.jar_path = jar_path if jar_path is not None else getattr(config, "GROBID_JAR_PATH", "")

        cmd = start_command if start_command is not None else getattr(config, "GROBID_START_COMMAND", "")
        if not cmd and self.jar_path:
            cmd = f"java -Xmx4g -jar {self.jar_path}"
        self.start_command = cmd

        self._pid_file = os.path.join(config.DATA_DIR, "grobid_process.pid")

    def _is_process_running(self) -> tuple[bool, Optional[int]]:
        """Check if a tracked background process is still alive."""
        if not os.path.exists(self._pid_file):
            return False, None
        try:
            with open(self._pid_file) as f:
                content = f.read().strip()
            if not content:
                return False, None
            pid = int(content)
            # Signal 0 checks if process exists without killing it
            os.kill(pid, 0)
            return True, pid
        except (ValueError, ProcessLookupError, PermissionError) as exc:
            if isinstance(exc, ProcessLookupError) or (isinstance(exc, OSError) and getattr(exc, "errno", None) == 3):
                try:
                    os.remove(self._pid_file)
                except OSError:
                    pass
            return False, None
        except Exception:
            return False, None

    # ─── Health Probing ───────────────────────────────────────────────────────

    def is_alive(self, timeout: float = 3.0, use_extractor_probe: bool = False) -> bool:
        """Probe the GROBID `/api/isalive` health endpoint.

        If `use_extractor_probe` is True, attempts to call `grobid_alive()`
        from `research_assistant.agents.agent1_extractor`.
        """
        if use_extractor_probe:
            try:
                from research_assistant.agents.agent1_extractor import grobid_alive
                return grobid_alive()
            except Exception as exc:
                logger.warning("Failed probing via agent1_extractor.grobid_alive: %s", exc)

        try:
            r = requests.get(f"{self.server_url}/api/isalive", timeout=timeout)
            return r.ok and "true" in r.text.lower()
        except requests.RequestException:
            return False

    def check_alive_via_extractor(self) -> bool:
        """Probe GROBID using `grobid_alive()` from `research_assistant.agents.agent1_extractor`."""
        from research_assistant.agents.agent1_extractor import grobid_alive
        return grobid_alive()

    # ─── URL / Host Helpers ───────────────────────────────────────────────────

    def get_port(self) -> int:
        """Extract the target port from server_url, defaulting to 8070."""
        parsed = urlparse(self.server_url)
        return parsed.port or 8070

    def is_local_target(self) -> bool:
        """Return True if server_url points to localhost or 127.0.0.1."""
        parsed = urlparse(self.server_url)
        hostname = (parsed.hostname or "").lower()
        return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "")

    # ─── Docker Environment & Discovery Checks ────────────────────────────────

    def check_docker(self) -> tuple[bool, str]:
        """Check if Docker CLI is installed and the Docker daemon is accessible."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False, "Docker executable not found in PATH. Please install Docker or start it manually."

        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=6,
            )
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out while checking daemon status."
        except Exception as exc:
            return False, f"Failed to execute docker: {exc}"

        if res.returncode == 0:
            return True, "Docker daemon is running and accessible."

        err = (res.stderr or res.stdout or "").strip()
        err_lower = err.lower()
        if "permission denied" in err_lower:
            return (
                False,
                "Docker socket permission denied. Check user access to /var/run/docker.sock or ~/.docker/run/docker.sock.",
            )
        if "cannot connect" in err_lower or "is the docker daemon running" in err_lower:
            return False, "Docker daemon is not running. Please start Docker Desktop."

        return False, f"Docker error: {err}"

    def find_active_container(self) -> Optional[dict]:
        """Discover any running or existing Docker container associated with GROBID."""
        docker_ok, _ = self.check_docker()
        if not docker_ok:
            return None

        # 1. Check self.container_name first
        try:
            res = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}\t{{.Name}}\t{{.State.Status}}\t{{.Config.Image}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split("\t")
                cid = parts[0][:12]
                name = parts[1].lstrip("/") if len(parts) > 1 else self.container_name
                status = parts[2].lower() if len(parts) > 2 else "unknown"
                image = parts[3] if len(parts) > 3 else ""
                return {"id": cid, "name": name, "status": status, "image": image}
        except Exception:
            pass

        # 2. Check running / existing containers for port mapping or grobid image
        port = self.get_port()
        try:
            res = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Image}}\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if res.returncode == 0:
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                # First pass: Running containers bound to port or running grobid
                for line in lines:
                    parts = line.split("\t")
                    cid = parts[0]
                    cnames = parts[1] if len(parts) > 1 else ""
                    cstate = parts[2].lower() if len(parts) > 2 else ""
                    cimage = parts[3] if len(parts) > 3 else ""
                    cports = parts[4] if len(parts) > 4 else ""

                    is_port_bound = f":{port}->" in cports or f":{port}/" in cports
                    is_grobid = "grobid" in cimage.lower() or "grobid" in cnames.lower() or cnames == self.container_name

                    if "running" in cstate:
                        if is_port_bound and is_grobid:
                            return {"id": cid, "name": cnames, "status": "running", "image": cimage, "is_grobid": True}
                        elif is_port_bound and not is_grobid:
                            # Port conflict: Container is bound to target port but is not GROBID
                            return {
                                "id": cid,
                                "name": cnames,
                                "status": "running",
                                "image": cimage,
                                "is_grobid": False,
                                "port_conflict": True,
                            }
                        elif is_grobid:
                            return {"id": cid, "name": cnames, "status": "running", "image": cimage, "is_grobid": True}

                # Second pass: Exited / stopped container matching container_name or grobid
                for line in lines:
                    parts = line.split("\t")
                    cid = parts[0]
                    cnames = parts[1] if len(parts) > 1 else ""
                    cstate = parts[2].lower() if len(parts) > 2 else ""
                    cimage = parts[3] if len(parts) > 3 else ""

                    if cnames == self.container_name or ("grobid" in cimage.lower() and "grobid" in cnames.lower()):
                        state = "running" if "running" in cstate else "exited"
                        return {"id": cid, "name": cnames, "status": state, "image": cimage, "is_grobid": True}
        except Exception:
            pass

        return None

    def get_container_info(self) -> dict:
        """Inspect the current container state for self.container_name or discovered container."""
        docker_ok, reason = self.check_docker()
        if not docker_ok:
            return {"exists": False, "status": "unavailable", "name": self.container_name, "detail": reason}

        # Check self.container_name
        try:
            res = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}\t{{.Config.Image}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split("\t")
                status = parts[0].lower()
                image = parts[1] if len(parts) > 1 else self.docker_image
                is_grobid = "grobid" in image.lower() or "grobid" in self.container_name.lower()
                return {
                    "exists": True,
                    "status": status,
                    "name": self.container_name,
                    "image": image,
                    "is_grobid": is_grobid,
                    "detail": f"Container '{self.container_name}' status: {status}",
                }
        except Exception as exc:
            return {"exists": False, "status": "error", "name": self.container_name, "detail": str(exc)}

        # If named container doesn't exist, check for active container
        active = self.find_active_container()
        if active:
            return {
                "exists": True,
                "status": active["status"],
                "name": active["name"],
                "image": active.get("image", ""),
                "is_grobid": active.get("is_grobid", True),
                "port_conflict": active.get("port_conflict", False),
                "detail": (
                    f"Port {self.get_port()} occupied by non-GROBID container '{active['name']}' ({active.get('image', '')})"
                    if active.get("port_conflict")
                    else f"Discovered container '{active['name']}' status: {active['status']}"
                ),
            }

        return {
            "exists": False,
            "status": "not_found",
            "name": self.container_name,
            "detail": f"Container '{self.container_name}' does not exist.",
        }

    def get_available_grobid_images(self) -> list[str]:
        """Return list of locally cached Docker images containing 'grobid'."""
        docker_ok, _ = self.check_docker()
        if not docker_ok:
            return []

        try:
            res = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if res.returncode == 0:
                images = [line.strip() for line in res.stdout.splitlines() if "grobid" in line.lower()]
                return [img for img in images if img and "<none>" not in img]
        except Exception:
            pass
        return []

    def get_container_logs(self, lines: int = 25) -> str:
        """Fetch recent logs from the GROBID container."""
        docker_ok, _ = self.check_docker()
        if not docker_ok:
            return "Docker not available to read logs."

        c_info = self.get_container_info()
        target_name = c_info.get("name") or self.container_name
        try:
            res = subprocess.run(
                ["docker", "logs", "--tail", str(lines), target_name],
                capture_output=True,
                text=True,
                timeout=6,
            )
            return (res.stdout + res.stderr).strip() or "No log output."
        except Exception as exc:
            return f"Failed to retrieve logs: {exc}"

    def get_manual_command(self) -> str:
        """Return a copy-pasteable terminal command to launch GROBID."""
        port = self.get_port()
        return f"docker run --rm -d --name {self.container_name} -p {port}:8070 {self.docker_image}"

    # ─── Status Assessment ────────────────────────────────────────────────────

    def check_status(self) -> GrobidStatus:
        """Evaluate the overall GROBID server status and runtime diagnostics."""
        manual_cmd = self.get_manual_command()
        alive = self.is_alive()

        docker_ok, docker_msg = self.check_docker()
        c_info = self.get_container_info() if docker_ok else {
            "exists": False,
            "status": "unavailable",
            "name": self.container_name,
            "detail": docker_msg,
        }
        resolved_name = c_info.get("name") or self.container_name
        c_status = c_info.get("status")

        if alive:
            return GrobidStatus(
                is_alive=True,
                state="RUNNING",
                message=f"GROBID server is active and responding at {self.server_url}.",
                server_url=self.server_url,
                docker_available=docker_ok,
                container_name=resolved_name,
                container_status=c_status,
                image=c_info.get("image") or self.docker_image,
                manual_command=manual_cmd,
            )

        # Server is not responding to /api/isalive
        if not self.is_local_target():
            return GrobidStatus(
                is_alive=False,
                state="ERROR",
                message=f"Remote GROBID server at {self.server_url} is unreachable.",
                server_url=self.server_url,
                docker_available=docker_ok,
                container_name=resolved_name,
                details="Local container management applies only to localhost instances.",
                manual_command=manual_cmd,
            )

        # Check port conflict first
        if c_info.get("port_conflict"):
            return GrobidStatus(
                is_alive=False,
                state="ERROR",
                message=f"Port {self.get_port()} is occupied by non-GROBID container '{resolved_name}' ({c_info.get('image', '')}).",
                server_url=self.server_url,
                docker_available=docker_ok,
                container_name=resolved_name,
                container_status=c_status,
                image=c_info.get("image"),
                details="A non-GROBID container is bound to this port. Stop that container or configure GROBID_SERVER with another port.",
                manual_command=manual_cmd,
            )

        # Check background process tracking
        proc_running, pid = self._is_process_running()
        if proc_running:
            return GrobidStatus(
                is_alive=False,
                state="STARTING",
                message=f"GROBID background process (PID {pid}) is running, waiting for service to initialize...",
                server_url=self.server_url,
                docker_available=docker_ok,
                container_name=resolved_name,
                container_status="running_process",
                image=c_info.get("image") or self.docker_image,
                manual_command=manual_cmd,
            )

        if not docker_ok and not self.start_command:
            return GrobidStatus(
                is_alive=False,
                state="STOPPED",
                message=f"GROBID server is stopped at {self.server_url}.",
                server_url=self.server_url,
                docker_available=False,
                container_name=resolved_name,
                details=docker_msg,
                manual_command=manual_cmd,
            )

        if c_status == "running":
            is_grobid = c_info.get("is_grobid", True)
            if is_grobid:
                return GrobidStatus(
                    is_alive=False,
                    state="STARTING",
                    message="Container is running, waiting for GROBID service to finish initializing JVM...",
                    server_url=self.server_url,
                    docker_available=True,
                    container_name=resolved_name,
                    container_status=c_status,
                    image=c_info.get("image") or self.docker_image,
                    manual_command=manual_cmd,
                )
            else:
                return GrobidStatus(
                    is_alive=False,
                    state="ERROR",
                    message=f"Container '{resolved_name}' is running on port {self.get_port()}, but is not a recognized GROBID server.",
                    server_url=self.server_url,
                    docker_available=True,
                    container_name=resolved_name,
                    container_status=c_status,
                    image=c_info.get("image"),
                    details="The running container does not match GROBID image signatures.",
                    manual_command=manual_cmd,
                )

        return GrobidStatus(
            is_alive=False,
            state="STOPPED",
            message=f"GROBID server is stopped at {self.server_url}.",
            server_url=self.server_url,
            docker_available=docker_ok,
            container_name=resolved_name,
            container_status=c_status,
            image=c_info.get("image") or self.docker_image,
            details=c_info.get("detail"),
            manual_command=manual_cmd,
        )

    # ─── Lifecycle Management ─────────────────────────────────────────────────

    def start_server(
        self,
        timeout: float = 25.0,
        poll_interval: float = 1.0,
    ) -> tuple[bool, str]:
        """Launch the GROBID server and poll until it becomes responsive."""
        if self.is_alive():
            return True, f"GROBID server is already running at {self.server_url}."

        if not self.is_local_target():
            return (
                False,
                f"Configured GROBID_SERVER ({self.server_url}) is a remote host. Cannot start remote servers locally.",
            )

        # 1. Custom process start command / JAR path if configured
        if self.start_command:
            logger.info("Launching GROBID via custom start command: %s", self.start_command)
            try:
                proc = subprocess.Popen(self.start_command, shell=True)
                os.makedirs(os.path.dirname(self._pid_file), exist_ok=True)
                with open(self._pid_file, "w") as f:
                    f.write(str(proc.pid))
            except Exception as exc:
                return False, f"Failed to execute start command: {exc}"
            return self._poll_health(timeout, poll_interval)

        # 2. Docker launch path
        docker_ok, docker_msg = self.check_docker()
        if not docker_ok:
            return (
                False,
                f"Cannot start GROBID: {docker_msg} Please start Docker Desktop or launch manually: {self.get_manual_command()}",
            )

        c_info = self.get_container_info()
        target_name = c_info.get("name") or self.container_name
        c_status = c_info.get("status")
        port = self.get_port()

        if c_info.get("port_conflict"):
            return (
                False,
                f"Port {port} is occupied by conflicting non-GROBID container '{target_name}' ({c_info.get('image', '')}). Please stop that container or configure GROBID_SERVER with a different port.",
            )

        if c_info.get("exists"):
            if c_status == "running":
                logger.info("GROBID container '%s' is already running; waiting for health probe.", target_name)
            elif c_status in ("exited", "created", "stopped"):
                logger.info("Starting existing stopped container '%s'...", target_name)
                res = subprocess.run(
                    ["docker", "start", target_name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if res.returncode != 0:
                    return False, f"Failed to start existing container '{target_name}': {res.stderr.strip()}"
            elif c_status == "paused":
                subprocess.run(["docker", "unpause", target_name], capture_output=True, text=True, timeout=6)
        else:
            # Container does not exist; choose best image
            available_images = self.get_available_grobid_images()
            image_to_use = self.docker_image
            if image_to_use not in available_images and available_images:
                tag = image_to_use.split(":")[-1] if ":" in image_to_use else ""
                matching_tag_images = [img for img in available_images if tag and img.endswith(f":{tag}")]
                if matching_tag_images:
                    image_to_use = matching_tag_images[0]
                else:
                    image_to_use = sorted(available_images, reverse=True)[0]
                logger.info("Default image '%s' not cached; using local cached image: %s", self.docker_image, image_to_use)

            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "-p",
                f"{port}:8070",
                image_to_use,
            ]
            logger.info("Executing: %s", " ".join(cmd))
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired:
                return False, "Docker run command timed out while starting GROBID."

            if res.returncode != 0:
                err = (res.stderr or res.stdout).strip()
                err_lower = err.lower()
                if "already in use" in err_lower or "conflict" in err_lower:
                    start_res = subprocess.run(
                        ["docker", "start", self.container_name],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if start_res.returncode != 0:
                        return False, f"Container conflict on '{self.container_name}': {err}"
                elif "unable to find image" in err_lower or "pull access denied" in err_lower:
                    return (
                        False,
                        f"GROBID image '{image_to_use}' not found locally. Please run: docker pull {image_to_use}",
                    )
                elif "port is already allocated" in err_lower or "address already in use" in err_lower:
                    return (
                        False,
                        f"Port {port} is already in use by another process. Stop the conflicting service or change GROBID_SERVER port.",
                    )
                else:
                    return False, f"Failed to launch GROBID container: {err}"

        return self._poll_health(timeout, poll_interval)

    def _poll_health(self, timeout: float, poll_interval: float) -> tuple[bool, str]:
        """Poll the /api/isalive endpoint until responsive or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_alive():
                logger.info("GROBID server is alive at %s.", self.server_url)
                return True, f"GROBID server successfully started and verified at {self.server_url}."
            time.sleep(poll_interval)

        # Check if container or process is still running even if JVM is initializing
        c_info = self.get_container_info()
        proc_running, _ = self._is_process_running()
        if (c_info.get("exists") and c_info.get("status") == "running") or proc_running:
            return (
                True,
                "GROBID container is running and initializing its JVM. It will become active shortly (click Check Status).",
            )

        logs = self.get_container_logs(lines=15)
        return False, f"GROBID failed to become ready within {timeout:.0f}s. Container logs:\n{logs}"

    def stop_server(self, timeout: float = 10.0) -> tuple[bool, str]:
        """Stop the GROBID server container or tracked process."""
        # Stop tracked process if PID file exists
        if os.path.exists(self._pid_file):
            target_pid = None
            try:
                with open(self._pid_file) as f:
                    content = f.read().strip()
                if content:
                    target_pid = int(content)
            except Exception:
                target_pid = None

            if target_pid is not None:
                try:
                    os.kill(target_pid, signal.SIGTERM)
                except Exception as exc:
                    logger.warning("Failed to terminate tracked process: %s", exc)

            if os.path.exists(self._pid_file):
                try:
                    os.remove(self._pid_file)
                except OSError:
                    pass
            return True, f"Stopped GROBID process with PID {target_pid}."

        docker_ok, docker_msg = self.check_docker()
        if not docker_ok:
            if self.is_alive():
                return (
                    False,
                    f"GROBID server is responding at {self.server_url}, but Docker is not accessible to stop it: {docker_msg}",
                )
            return True, "GROBID server is already stopped (Docker not available)."

        c_info = self.get_container_info()
        target_name = c_info.get("name") or self.container_name

        if not c_info.get("exists"):
            if self.is_alive():
                return (
                    False,
                    f"GROBID server is running at {self.server_url} but no container was found. It may be running as an external service.",
                )
            return True, f"GROBID container '{target_name}' does not exist and server is stopped."

        if c_info.get("status") != "running":
            return True, f"GROBID container '{target_name}' is already stopped."

        try:
            res = subprocess.run(
                ["docker", "stop", "-t", str(int(timeout)), target_name],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            if res.returncode == 0:
                logger.info("Stopped GROBID container '%s'.", target_name)
                return True, f"GROBID container '{target_name}' stopped successfully."
            return False, f"Failed to stop container '{target_name}': {res.stderr.strip()}"
        except Exception as exc:
            return False, f"Error stopping GROBID container: {exc}"

    def restart_server(self, timeout: float = 25.0) -> tuple[bool, str]:
        """Stop and restart the GROBID server."""
        self.stop_server()
        time.sleep(1.0)
        return self.start_server(timeout=timeout)

    def run_action(self, action: str, **kwargs) -> tuple[bool, str, GrobidStatus]:
        """Execute a lifecycle action: 'status', 'start', 'stop', 'restart'."""
        act = (action or "").lower().strip()
        if act == "start":
            ok, msg = self.start_server(**kwargs)
        elif act == "stop":
            ok, msg = self.stop_server(**kwargs)
        elif act == "restart":
            ok, msg = self.restart_server(**kwargs)
        elif act in ("status", "check"):
            st = self.check_status()
            return True, st.message, st
        else:
            return False, f"Unknown action: '{action}'.", self.check_status()
        return ok, msg, self.check_status()


# Aliases
GrobidController = GrobidManager
GrobidAgent = GrobidManager


# ─── Module-level Helper Functions ────────────────────────────────────────────

def is_grobid_alive(timeout: float = 3.0, use_extractor: bool = False) -> bool:
    """Check if GROBID is running by calling grobid_alive() from agent1_extractor or querying /api/isalive."""
    return GrobidManager().is_alive(timeout=timeout, use_extractor_probe=use_extractor)


def check_grobid_status() -> GrobidStatus:
    """Get the current GrobidStatus."""
    return GrobidManager().check_status()


def start_grobid(timeout: float = 25.0) -> tuple[bool, str]:
    """Start the GROBID server."""
    return GrobidManager().start_server(timeout=timeout)


def stop_grobid(timeout: float = 10.0) -> tuple[bool, str]:
    """Stop the GROBID server."""
    return GrobidManager().stop_server(timeout=timeout)


def restart_grobid(timeout: float = 25.0) -> tuple[bool, str]:
    """Restart the GROBID server."""
    return GrobidManager().restart_server(timeout=timeout)


def main() -> None:
    """CLI utility entry point for GrobidManager."""
    import sys

    manager = GrobidManager()
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "status":
        status = manager.check_status()
        print(f"Status: {status.state}")
        print(f"Alive:  {status.is_alive}")
        print(f"URL:    {status.server_url}")
        print(f"Detail: {status.message}")
        if status.details:
            print(f"Diag:   {status.details}")
    elif action == "start":
        ok, msg = manager.start_server()
        print(f"Result ({ok}): {msg}")
        sys.exit(0 if ok else 1)
    elif action == "stop":
        ok, msg = manager.stop_server()
        print(f"Result ({ok}): {msg}")
        sys.exit(0 if ok else 1)
    elif action == "restart":
        ok, msg = manager.restart_server()
        print(f"Result ({ok}): {msg}")
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown action: {action}. Choose from status, start, stop, restart.")
        sys.exit(2)


if __name__ == "__main__":
    main()
