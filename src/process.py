"""
Process Management for External Training Execution
Handles subprocess lifecycle, logging, and UI communication
"""

import json
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
            msg = "[FLUX-TRAIN] Warning: Process already running. Stop it first."
            print(msg)
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": msg})
                except Exception:
                    pass
            return

        self.stop_event.clear()
        self.log_lines = []

        # Environment setup for Ampere GPU (RTX 3060 Ti, RTX 4000 series, etc.)
        env = os.environ.copy()
        env["ACCELERATE_MIXED_PRECISION"] = "bf16"
        env["PYTHONIOENCODING"] = "utf-8"  # CRITICAL for Windows console output

        # === FIX: RELATIVE PATH STRATEGY for reliable 'library' module discovery ===
        # Problem: accelerate may spawn child processes that ignore/lose PYTHONPATH
        # Solution: Run process from sd-scripts directory with relative script name
        # This ensures Python's working directory is where 'library' module exists
        
        script_dir = cwd  # Default fallback
        script_name = None
        
        # 1. Try to find sd-scripts directory more reliably
        # First, extract script name and check if full path is provided
        for arg in cmd_list:
            if isinstance(arg, str) and arg.endswith(".py"):
                script_name = os.path.basename(arg)
                
                # If full path provided and exists, use its directory
                if os.path.exists(arg):
                    script_dir = os.path.dirname(os.path.abspath(arg))
                    break
        
        if script_name and script_dir == cwd:
            # Script name is relative - search for sd-scripts location
            possible_paths = [
                cwd,  # Assume cwd is sd-scripts
                os.path.join(cwd, "sd-scripts"),  # Or in a subdirectory
                os.path.join(cwd, "kohya_ss", "sd-scripts"),  # Alternative structure
                os.path.join(cwd, "kohya_train", "kohya_ss", "sd-scripts"),  # Common ComfyUI layout
                os.path.join(cwd, "custom_nodes", "sd-scripts"),  # In custom_nodes
                os.path.join(cwd, "..", "sd-scripts"),  # One level up
            ]
            
            for possible_path in possible_paths:
                candidate = os.path.join(possible_path, script_name)
                if os.path.exists(candidate):
                    script_dir = os.path.abspath(possible_path)
                    # Update cmd_list with full path to ensure it's found
                    for i, arg in enumerate(cmd_list):
                        if arg.endswith(".py") and not os.path.isabs(arg):
                            cmd_list[i] = candidate
                    break
            else:
                # Last resort: check for 'library' folder in possible paths
                for possible_path in possible_paths:
                    if os.path.exists(os.path.join(possible_path, "library")):
                        script_dir = os.path.abspath(possible_path)
                        # Update cmd_list with full path
                        for i, arg in enumerate(cmd_list):
                            if arg.endswith(".py"):
                                cmd_list[i] = os.path.join(script_dir, script_name)
                        break
        
        # 2. Set up PYTHONPATH as fallback (belt and suspenders approach)
        current_pythonpath = env.get("PYTHONPATH", "")
        
        # Add both script_dir and cwd, with script_dir first (highest priority)
        pythonpath_parts = [script_dir, cwd, current_pythonpath]
        # Filter out empty strings and duplicates
        pythonpath_parts = [p for p in pythonpath_parts if p]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        
        # Debug: Log the working directory and PYTHONPATH (helps troubleshooting)
        debug_lines = [
            f"Working Dir: {script_dir}",
            f"PYTHONPATH: {env['PYTHONPATH']}",
            f"Looking for: {os.path.join(script_dir, 'library')}",
        ]
        
        for debug_msg in debug_lines:
            print(f"[DEBUG] {debug_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"DEBUG: {debug_msg}"})
                except Exception:
                    pass
        # ========================================================================
        
        # CRITICAL: Verify script exists before attempting to run
        script_to_run = None
        for arg in cmd_list:
            if isinstance(arg, str) and arg.endswith(".py"):
                script_to_run = arg
                break
        
        if script_to_run:
            # If relative path, try to resolve it
            if not os.path.isabs(script_to_run):
                script_to_run = os.path.join(script_dir, script_to_run)
            
            if not os.path.exists(script_to_run):
                error_msg = f"CRITICAL ERROR: Training script not found: {script_to_run}"
                print(f"[FLUX-TRAIN] {error_msg}")
                print(f"[FLUX-TRAIN] Search paths tried:")
                for p in [cwd, script_dir]:
                    print(f"  - {p}")
                
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": error_msg})
                        PromptServer.instance.send_sync("flux_train_log", {"line": f"Expected at: {script_to_run}"})
                        PromptServer.instance.send_sync("flux_train_log", {"line": "Possible fixes:"})
                        PromptServer.instance.send_sync("flux_train_log", {"line": "1. Download sd-scripts from: https://github.com/kohya-ss/sd-scripts"})
                        PromptServer.instance.send_sync("flux_train_log", {"line": f"2. Place in: {os.path.join(cwd, 'sd-scripts')}"})
                        PromptServer.instance.send_sync("flux_train_log", {"line": "3. OR update sd-scripts path in configurator node"})
                    except Exception:
                        pass
                
                raise FileNotFoundError(f"Training script not found: {script_to_run}")

        # Windows-specific: Create new console group to allow clean termination
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0

        try:
            # Launch subprocess with isolated I/O
            # CRITICAL: Run from sd-scripts directory so relative script name resolves
            # and so Python can find 'library' module in current working directory
            self.process = subprocess.Popen(
                cmd_list,
                cwd=script_dir,  # This MUST be sd-scripts directory!
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
                universal_newlines=True,
                creationflags=creationflags
            )

            # Send startup messages
            startup_msg = "--- TRAINING PROCESS STARTED ---"
            print(f"[FLUX-TRAIN] {startup_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": startup_msg})
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"Working Dir: {script_dir}"})
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"PYTHONPATH: {env.get('PYTHONPATH', 'not set')}"})
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"Command: {' '.join(cmd_list)}"})
                except Exception as e:
                    print(f"[FLUX-TRAIN] Warning: Could not send startup message: {e}")

            # Start log reader in daemon thread
            log_thread = threading.Thread(target=self._log_reader, daemon=True)
            log_thread.start()
            
        except Exception as e:
            err_msg = f"FAILED TO START PROCESS: {str(e)}"
            print(f"[FLUX-TRAIN] {err_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": err_msg})
                except Exception:
                    pass
            raise

    def _log_reader(self) -> None:
        """
        Reads stdout from subprocess and forwards to ComfyUI browser interface.
        Runs in separate thread to avoid blocking main ComfyUI process.
        """
        try:
            while self.process and self.process.poll() is None:
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    
                    # If line is empty and process ended, break
                    if not line and self.process.poll() is not None:
                        break
                        
                    if line:
                        clean_line = line.rstrip("\n\r")
                        self.log_lines.append(clean_line)

                        # Limit in-memory log size
                        if len(self.log_lines) > 500:
                            self.log_lines = self.log_lines[-300:]

                        # Print to server console (for debugging)
                        print(f"[FLUX-TRAIN] {clean_line}")

                        # Send to browser UI via WebSocket
                        if PromptServer:
                            try:
                                PromptServer.instance.send_sync(
                                    "flux_train_log",
                                    {"line": clean_line}
                                )
                            except Exception as e:
                                print(f"[FLUX-TRAIN] Warning: Could not send log: {e}")
                    else:
                        time.sleep(0.01)  # Small sleep if no data
        except Exception as e:
            err_msg = f"Error in log reader: {e}"
            print(f"[FLUX-TRAIN] {err_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": err_msg})
                except Exception:
                    pass

                time.sleep(0.01)  # Small sleep to prevent busy-waiting

        except Exception as e:
            error_msg = f"[FLUX-TRAIN] Error in log reader: {e}"
            print(error_msg)
            if PromptServer:
                try:
                    PromptServer.instance.send_sync(
                        "flux_train_log",
                        {"line": error_msg}
                    )
                except Exception:
                    pass

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
            cmd_args: Command as JSON list (from Flux2_8GB_Configurator) or legacy string format
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
            # CRITICAL FIX: Parse command as JSON (preserves Windows paths)
            # This avoids shlex.split() mangling backslashes on Windows
            cmd_list = []
            
            try:
                # Try JSON format first (v1.2+ format with path preservation)
                cmd_list = json.loads(cmd_args)
                if not isinstance(cmd_list, list):
                    raise ValueError("JSON is not a list")
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: Parse as string for backward compatibility
                print(f"[FLUX-TRAIN] Warning: Using legacy string parsing (may fail on Windows paths): {e}")
                cmd_list = shlex.split(cmd_args)

            if not cmd_list or not isinstance(cmd_list, list):
                return ("❌ Error: Invalid command format",)

            # Infer working directory from script path or common locations
            cwd = os.getcwd()
            for arg in cmd_list:
                if isinstance(arg, str) and arg.endswith(".py"):
                    if os.path.exists(arg):
                        cwd = os.path.dirname(os.path.abspath(arg)) or os.getcwd()
                    else:
                        # Try to find sd-scripts directory
                        for search_path in [
                            os.getcwd(),
                            os.path.join(os.getcwd(), "sd-scripts"),
                            os.path.join(os.getcwd(), "kohya_ss", "sd-scripts"),
                            os.path.join(os.getcwd(), "kohya_train", "kohya_ss", "sd-scripts"),
                            os.path.join(os.getcwd(), "..", "sd-scripts"),
                        ]:
                            if os.path.exists(os.path.join(search_path, arg)):
                                cwd = search_path
                                break
                    break

            manager.start_training(cmd_list, cwd=cwd)
            return ("✅ Training Started! Monitor console for progress.",)

        except FileNotFoundError as e:
            error_msg = f"❌ File Error: {str(e)}\nCheck console for details."
            print(f"[FLUX-TRAIN] {error_msg}")
            return (error_msg,)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[FLUX-TRAIN] Full error:\n{error_detail}")
            return (f"❌ Error: {str(e)}\nCheck console for traceback.",)


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
