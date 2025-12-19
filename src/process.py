"""
Process Management for External Training Execution
Handles subprocess lifecycle, logging, and UI communication
"""

import threading
import subprocess
import os
import shlex
import time
from typing import Optional, List

# ComfyUI imports (graceful fallback if not available)
try:
    from server import PromptServer
except ImportError:
    PromptServer = None


class TrainingProcessManager:
    """
    Singleton for managing background training processes.
    Prevents UI freezing by running training in a separate system process.
    Streams logs back to ComfyUI console.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.log_lines: List[str] = []

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def start_training(self, cmd_list: List[str], cwd: str) -> None:
        """
        Start training process in isolated subprocess.
        
        Args:
            cmd_list: Command as list of strings
            cwd: Working directory for the process
            
        Raises:
            RuntimeError: If training is already running
        """
        if self.process and self.process.poll() is None:
            raise RuntimeError("Training is already running!")

        self.stop_event.clear()
        self.log_lines = []

        # Environment setup for Ampere GPU (RTX 3060 Ti, RTX 4000 series, etc.)
        env = os.environ.copy()
        env["ACCELERATE_MIXED_PRECISION"] = "bf16"

        # Windows-specific: Create new console group to allow clean termination
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0

        # Launch subprocess with isolated I/O
        self.process = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags
        )

        # Start log reader in daemon thread
        log_thread = threading.Thread(target=self._log_reader, daemon=True)
        log_thread.start()

    def _log_reader(self) -> None:
        """
        Reads stdout from subprocess and forwards to ComfyUI console.
        Runs in separate thread to avoid blocking.
        """
        try:
            while self.process and self.process.poll() is None:
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        clean_line = line.strip()
                        self.log_lines.append(clean_line)

                        # Limit in-memory log size
                        if len(self.log_lines) > 500:
                            self.log_lines = self.log_lines[-300:]

                        # Print to console (visible to user)
                        print(f"[FLUX-TRAIN] {clean_line}")

                        # Optional: Send to PromptServer if available
                        if PromptServer:
                            try:
                                PromptServer.instance.send_sync(
                                    "flux.training.log",
                                    {"message": clean_line, "timestamp": time.time()}
                                )
                            except Exception:
                                pass  # Silently fail if server not available
                time.sleep(0.01)  # Small sleep to prevent busy-waiting
        except Exception as e:
            print(f"[FLUX-TRAIN] Error in log reader: {e}")

    def stop_training(self) -> None:
        """Gracefully stop training process."""
        if not self.process:
            return

        print("[FLUX-TRAIN] Stopping training process...")
        
        try:
            # Try soft termination first (SIGINT on Unix, CtrlC on Windows)
            if os.name == 'nt':
                # Windows: send Ctrl+C to process group
                os.kill(self.process.pid, 9)  # SIGKILL on Windows
            else:
                # Unix: send SIGINT
                self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[FLUX-TRAIN] Process did not terminate gracefully, killing...")
                self.process.kill()

        except Exception as e:
            print(f"[FLUX-TRAIN] Error stopping process: {e}")
        finally:
            self.process = None

    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self.process is not None and self.process.poll() is None

    def get_last_logs(self, n: int = 50) -> str:
        """Get last n lines from log buffer."""
        tail = self.log_lines[-n:]
        return "\n".join(tail)


# --- ComfyUI Nodes ---

class Flux2_Runner:
    """
    Runs external training process using command from Flux2_8GB_Configurator.
    
    This node is marked as OUTPUT_NODE to ensure it runs as a workflow terminator.
    The actual training happens in a subprocess, keeping ComfyUI responsive.
    """
    
    CATEGORY = "Flux2/Training"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run_training"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmd_args": ("STRING", {"forceInput": True, "multiline": True}),
                "trigger": ("BOOLEAN", {"default": False, "label": "RUN TRAINING"}),
            }
        }

    def run_training(self, cmd_args: str, trigger: bool):
        """
        Start training if trigger is True.
        
        Args:
            cmd_args: Command string from Flux2_8GB_Configurator
            trigger: Boolean flag to start training
            
        Returns:
            Tuple with status message
        """
        if not trigger:
            return ("⏸️ Waiting for trigger (set to True to start)...",)

        manager = TrainingProcessManager.get_instance()

        # Check if already running
        if manager.is_running():
            return ("⚠️ Training already running! Cancel first.",)

        try:
            # Parse command string safely
            cmd_list = shlex.split(cmd_args)

            # Infer working directory from script path
            # Usually: accelerate launch ... /path/to/script.py ...
            cwd = os.getcwd()
            for i, arg in enumerate(cmd_list):
                if arg.endswith(".py"):
                    cwd = os.path.dirname(arg) or os.getcwd()
                    break

            manager.start_training(cmd_list, cwd=cwd)
            return ("✅ Training Started! Monitor console for progress.",)

        except Exception as e:
            return (f"❌ Error: {str(e)}",)


class Flux2_Stopper:
    """
    Emergency stop node to terminate running training process.
    Marked as OUTPUT_NODE to ensure execution.
    """
    
    CATEGORY = "Flux2/Training"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "stop_process"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stop": ("BOOLEAN", {"default": False, "label": "STOP TRAINING"})
            }
        }

    def stop_process(self, stop: bool):
        """Stop training if stop is True."""
        manager = TrainingProcessManager.get_instance()
        
        if stop and manager.is_running():
            manager.stop_training()
            return ("🛑 Stop signal sent",)
        elif not manager.is_running():
            return ("ℹ️ No training running",)
        else:
            return ("⏸️ Stop disabled",)
