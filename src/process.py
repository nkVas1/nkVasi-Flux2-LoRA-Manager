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

# Virtual environment management
from .venv_manager import StandalonePackageManager, ensure_training_packages

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
        
        # === PRE-FLIGHT ENVIRONMENT CHECK ===
        try:
            from .environment_checker import EnvironmentChecker
            
            print("[FLUX-TRAIN] Running environment check...")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": "Running environment check..."})
                except Exception:
                    pass
            
            all_ok, messages = EnvironmentChecker.run_full_check()
            
            for msg in messages:
                print(f"[FLUX-TRAIN] {msg}")
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": msg})
                    except Exception:
                        pass
            
            if not all_ok:
                warning = "⚠ Environment check found issues - training may fail"
                print(f"[FLUX-TRAIN] {warning}")
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": warning})
                    except Exception:
                        pass
        except ImportError:
            print("[FLUX-TRAIN] Environment checker not available, skipping pre-flight check")
        # =====================================
        
        # === TRAINING PACKAGES SETUP ===
        print("[FLUX-TRAIN] Ensuring training packages are installed...")
        if PromptServer:
            try:
                PromptServer.instance.send_sync("flux_train_log", {"line": "Checking training packages..."})
            except Exception:
                pass

        # Get plugin base directory
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        try:
            # Ensure packages are installed
            success, msg, libs_path = ensure_training_packages(plugin_dir)
            
            if not success:
                error_msg = f"Failed to setup training packages: {msg}"
                print(f"[FLUX-TRAIN] {error_msg}")
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": error_msg})
                        PromptServer.instance.send_sync("flux_train_log", {"line": "Try: Delete training_libs folder and restart"})
                    except Exception:
                        pass
                # Don't raise - fallback to system packages
                print("[FLUX-TRAIN] Falling back to system Python packages")
            else:
                print(f"[FLUX-TRAIN] ✓ {msg}")
                print(f"[FLUX-TRAIN] Using packages from: {libs_path}")
                
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": f"✓ {msg}"})
                    except Exception:
                        pass

        except Exception as e:
            error_msg = f"Package manager error: {e}"
            print(f"[FLUX-TRAIN] {error_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": error_msg})
                except Exception:
                    pass
            print("[FLUX-TRAIN] Continuing with system Python packages")
        # ===================================
        
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
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output for real-time logs
        
        # CRITICAL FIX: Disable problematic modules for embedded Python
        # Triton/bitsandbytes require Python.h which embedded Python lacks
        env["BITSANDBYTES_NOWELCOME"] = "1"
        env["DISABLE_TRITON"] = "1"
        env["DISABLE_BITSANDBYTES_WARN"] = "1"
        
        # Force use of standard quantization instead of triton-based
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Prevent diffusers from trying to compile C extensions
        env["DIFFUSERS_DISABLE_TELEMETRY"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        # Modify PYTHONPATH to use training_libs if available
        try:
            manager = StandalonePackageManager(plugin_dir)
            env = manager.get_modified_env(env)
            print(f"[FLUX-TRAIN] Updated PYTHONPATH for training packages")
        except Exception as e:
            print(f"[FLUX-TRAIN] Could not update PYTHONPATH: {e}")


        # === ULTIMATE FIX: Wrapper script to guarantee library module discovery ===
        # Problem: accelerate subprocess loses parent's sys.path and cwd context
        # Solution: Create wrapper that explicitly adds sd-scripts to sys.path BEFORE imports
        
        script_dir = cwd  # Default fallback
        script_name = None
        original_script_path = None
        # 1. Find sd-scripts directory and original script path
        # First, extract script name and check if full path is provided
        for arg in cmd_list:
            if isinstance(arg, str) and arg.endswith(".py"):
                script_name = os.path.basename(arg)
                
                # If full path provided and exists, use its directory
                if os.path.exists(arg):
                    script_dir = os.path.dirname(os.path.abspath(arg))
                    original_script_path = os.path.abspath(arg)
                    break
        
        if script_name and not original_script_path:
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
                    original_script_path = os.path.abspath(candidate)
                    # Update cmd_list with full path to ensure it's found
                    for i, arg in enumerate(cmd_list):
                        if arg.endswith(".py") and not os.path.isabs(arg):
                            cmd_list[i] = original_script_path
                    break
            else:
                # Last resort: check for 'library' folder in possible paths
                for possible_path in possible_paths:
                    if os.path.exists(os.path.join(possible_path, "library")):
                        script_dir = os.path.abspath(possible_path)
                        original_script_path = os.path.join(script_dir, script_name)
                        # Update cmd_list with full path
                        for i, arg in enumerate(cmd_list):
                            if arg.endswith(".py"):
                                cmd_list[i] = original_script_path
                        break
        
        # 2. Prepare PYTHONPATH with ABSOLUTE paths
        script_dir_abs = os.path.abspath(script_dir)
        cwd_abs = os.path.abspath(cwd)
        current_pythonpath = env.get("PYTHONPATH", "")
        
        # Build PYTHONPATH with absolute paths only
        pythonpath_parts = [script_dir_abs]
        
        # Add ComfyUI root for other imports
        if cwd_abs != script_dir_abs:
            pythonpath_parts.append(cwd_abs)
        
        # Preserve existing PYTHONPATH
        if current_pythonpath:
            pythonpath_parts.extend(current_pythonpath.split(os.pathsep))
        
        # Remove duplicates while preserving order
        seen = set()
        pythonpath_final = []
        for p in pythonpath_parts:
            if p and p not in seen:
                seen.add(p)
                pythonpath_final.append(p)
        
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_final)
        
        # 3. Create wrapper script that adds paths to sys.path BEFORE imports
        # This guarantees that when accelerate subprocess runs, it can find 'library'
        wrapper_script_path = None
        if original_script_path:
            # Convert paths to forward slashes to avoid escape sequence issues
            script_dir_forward = script_dir_abs.replace('\\', '/')
            plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plugin_src_forward = os.path.join(plugin_dir, 'src').replace('\\', '/')
            original_script_forward = original_script_path.replace('\\', '/')
            
            wrapper_content = f'''import sys
import os

# === STEP 0: IMMEDIATE emergency triton blocking (BEFORE ANY IMPORTS) ===
# torch._dynamo.utils tries to import triton.language.dtype on torch import
# We must intercept this at the earliest possible moment
class _EmergencyTriton:
    """Emergency fake triton - blocks torch._dynamo triton access."""
    class _Language:
        dtype = type
    
    language = _Language()
    compiler = None
    runtime = None
    
    def __getattr__(self, name):
        return self
    
    def __call__(self, *args, **kwargs):
        def wrapper(f):
            return f
        return wrapper

sys.modules['triton'] = _EmergencyTriton()
sys.modules['triton.language'] = _EmergencyTriton._Language()
print("[WRAPPER] ⚡ Emergency triton blocker loaded (before torch import)")

# Set environment variables to disable all triton/compile features
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["DISABLE_TRITON"] = "1"
os.environ["TRITON_ENABLED"] = "0"

# === STEP 1: Prioritize training_libs in sys.path ===
# This MUST be FIRST to override system packages
training_libs = r"{script_dir_abs}".replace("\\\\", "/") + "/../../../custom_nodes/ComfyUI-Flux2-LoRA-Manager/training_libs"
training_libs = os.path.normpath(training_libs)

if os.path.exists(training_libs):
    # Insert at position 0 (highest priority)
    sys.path.insert(0, training_libs)
    print(f"[WRAPPER] ✓ Training libs prioritized: {{training_libs}}")

# === STEP 2: Add sd-scripts to sys.path ===
sys.path.insert(0, r"{script_dir_forward}")

# === STEP 3: Enhanced import blocker system ===
try:
    plugin_src = r"{plugin_src_forward}"
    if plugin_src not in sys.path:
        sys.path.insert(0, plugin_src)
    
    from import_blocker import install_import_blockers, verify_blockers_active
    install_import_blockers()
    
    if verify_blockers_active():
        print("[WRAPPER] ✓ Import protection system activated")
    else:
        print("[WRAPPER] ⚠ Import blockers partially active")
        
except Exception as e:
    print(f"[WRAPPER] WARNING: Import blocker setup failed: {{e}}")
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# === STEP 4: Verify library module ===
library_path = os.path.join(r"{script_dir_forward}", "library")
if not os.path.exists(library_path):
    print(f"[WRAPPER] ERROR: library not found at {{library_path}}")
    sys.exit(1)

print(f"[WRAPPER] Added sd-scripts to sys.path: {script_dir_forward}")
print(f"[WRAPPER] library module accessible")

# === STEP 5: Debug - check transformers source ===
try:
    import transformers
    print(f"[WRAPPER] transformers version: {{transformers.__version__}}")
    print(f"[WRAPPER] transformers from: {{transformers.__file__}}")
except ImportError as e:
    print(f"[WRAPPER] WARNING: transformers import failed: {{e}}")

# === STEP 6: Execute training script ===
try:
    script_path = r"{original_script_forward}"
    with open(script_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), script_path, "exec")
        exec(code)
except Exception as e:
    print(f"[WRAPPER] Training script error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
            
            # Write wrapper to temporary file
            import tempfile
            try:
                wrapper_fd, wrapper_script_path = tempfile.mkstemp(suffix=".py", prefix="flux_train_wrapper_", text=True)
                with os.fdopen(wrapper_fd, "w", encoding="utf-8") as f:
                    f.write(wrapper_content)
                
                # Replace original script with wrapper in command list
                for i, arg in enumerate(cmd_list):
                    if arg == original_script_path or (isinstance(arg, str) and arg.endswith("flux_train_network.py")):
                        cmd_list[i] = wrapper_script_path
                        break
                
                debug_msg = f"Created wrapper script: {wrapper_script_path}"
                print(f"[DEBUG] {debug_msg}")
                if PromptServer:
                    try:
                        PromptServer.instance.send_sync("flux_train_log", {"line": f"DEBUG: {debug_msg}"})
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARNING] Failed to create wrapper script: {e}")
                wrapper_script_path = None
        
        # Store wrapper path for cleanup later
        self._wrapper_script = wrapper_script_path
        
        # 4. Debug logging
        debug_lines = [
            f"Working Dir: {script_dir_abs}",
            f"PYTHONPATH: {env['PYTHONPATH']}",
            f"Library path: {os.path.join(script_dir_abs, 'library')}",
            f"Wrapper script: {wrapper_script_path if wrapper_script_path else 'not created'}",
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
            # CRITICAL: Run from sd-scripts directory so Python can find 'library' module
            self.process = subprocess.Popen(
                cmd_list,
                cwd=script_dir_abs,  # Use absolute path - MUST be sd-scripts directory!
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
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"Working Dir: {script_dir_abs}"})
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"PYTHONPATH: {env.get('PYTHONPATH', 'not set')}"})
                    PromptServer.instance.send_sync("flux_train_log", {"line": f"Wrapper: {self._wrapper_script if hasattr(self, '_wrapper_script') and self._wrapper_script else 'N/A'}"})
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
        finally:
            # Clean up wrapper script if it was created
            if hasattr(self, '_wrapper_script') and self._wrapper_script:
                try:
                    if os.path.exists(self._wrapper_script):
                        os.remove(self._wrapper_script)
                        print(f"[DEBUG] Cleaned up wrapper: {self._wrapper_script}")
                except Exception as e:
                    print(f"[DEBUG] Failed to clean wrapper: {e}")

            # Send completion message
            completion_msg = "--- TRAINING PROCESS COMPLETED ---"
            print(f"[FLUX-TRAIN] {completion_msg}")
            if PromptServer:
                try:
                    PromptServer.instance.send_sync("flux_train_log", {"line": completion_msg})
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
    
    **CRITICAL**: This node is OUTPUT_NODE but ONLY executes when:
    1. trigger=True AND
    2. Either:
       a) No training is running (starts new), OR
       b) Training is already running (returns status)
    
    This design prevents infinite loops while maintaining single-click training start.
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
                "trigger": ("BOOLEAN", {"default": False, "label": "🚀 START TRAINING"}),
            }
        }

    def run_training(self, cmd_args: str, trigger: bool):
        """
        Start training if trigger is True.
        
        **CRITICAL LOGIC**:
        - trigger=False → Return immediately (no execution, no loop)
        - trigger=True + no process running → Start training
        - trigger=True + process running → Return status (no restart)
        
        Args:
            cmd_args: Command as JSON list (from Flux2_8GB_Configurator) or legacy string format
            trigger: Boolean flag to start training
            
        Returns:
            Tuple with status message
        """
        # === KEY FIX FOR INFINITE LOOP ===
        # If trigger is False, return IMMEDIATELY without ANY side effects
        # This prevents the node from executing on every workflow refresh
        if not trigger:
            manager = TrainingProcessManager.get_instance()
            if manager.is_running():
                # Process is running, user just hasn't set trigger=True again
                status = "⏳ Training in progress... Set trigger=True to get status"
                return (status,)
            else:
                # No process running
                return ("⏸️ Ready. Set trigger=True to start training",)
        
        # === TRIGGER IS TRUE - EXECUTE ===
        manager = TrainingProcessManager.get_instance()

        # If already running, just return status (don't restart)
        if manager.is_running():
            # Get current training progress from logs
            last_logs = manager.get_last_logs(5)
            status = f"⏳ Training running:\n{last_logs}"
            return (status,)

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
            return ("✅ Training Started! Set trigger=False to monitor, set trigger=True again to check status",)

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
    
    **CRITICAL**: Only executes when stop=True.
    When stop=False, returns immediately without side effects (prevents loops).
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
                "stop": ("BOOLEAN", {"default": False, "label": "🛑 STOP TRAINING"})
            }
        }

    def stop_process(self, stop: bool):
        """
        Stop training process when stop=True.
        
        **CRITICAL LOGIC**:
        - stop=False → Return status without execution
        - stop=True → Stop training immediately
        
        Args:
            stop: Boolean flag to stop training
            
        Returns:
            Tuple with status message
        """
        manager = TrainingProcessManager.get_instance()
        
        # === KEY FIX FOR INFINITE LOOP ===
        # Always return immediately with status, regardless of stop value
        if not stop:
            # User has not clicked stop - return current status
            if manager.is_running():
                return ("⏳ Training running. Set stop=True to stop",)
            else:
                return ("ℹ️ No training running",)
        
        # === STOP=TRUE - EXECUTE STOP ===
        if manager.is_running():
            manager.stop_training()
            return ("✅ Stop signal sent. Process terminating...",)
        else:
            return ("ℹ️ No training running to stop",)
