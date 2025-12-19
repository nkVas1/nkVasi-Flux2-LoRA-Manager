/**
 * Advanced Progress Tracker for Flux2 Training - Mega Panel UI
 * Features: Real-time progress bars, package installation tracking, ETA estimation
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("%c[FLUX Progress] Loading Extension v2.0", "color: cyan; font-weight: bold; font-size: 16px;");

// Progress tracking state
const state = {
    packages: {
        total: 9,
        current: 0,
        currentName: "",
        startTime: null
    },
    training: {
        totalSteps: 0,
        currentStep: 0,
        startTime: null,
        lastLoss: null
    }
};

app.registerExtension({
    name: "Flux2.LoRA.ProgressTracker.Mega",
    
    async setup() {
        console.log("[FLUX Progress] Initializing UI...");
        
        // Remove old panel if exists
        const oldPanel = document.getElementById("flux-progress-mega-panel");
        if (oldPanel) oldPanel.remove();
        
        // ======================================================================
        // CREATE MEGA PROGRESS PANEL (Always Visible During Operations)
        // ======================================================================
        
        const megaPanel = document.createElement("div");
        megaPanel.id = "flux-progress-mega-panel";
        
        Object.assign(megaPanel.style, {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "600px",
            maxHeight: "400px",
            background: "linear-gradient(135deg, rgba(20, 20, 40, 0.98), rgba(40, 20, 60, 0.98))",
            color: "#fff",
            fontFamily: "'Segoe UI', Arial, sans-serif",
            fontSize: "14px",
            padding: "25px",
            borderRadius: "16px",
            border: "3px solid #00ffaa",
            boxShadow: "0 20px 60px rgba(0, 255, 170, 0.4), inset 0 0 30px rgba(0, 255, 170, 0.1)",
            zIndex: "99999",
            display: "none",
            backdropFilter: "blur(20px)",
            animation: "fadeIn 0.3s ease"
        });
        
        megaPanel.innerHTML = `
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; transform: translate(-50%, -45%); }
                    to { opacity: 1; transform: translate(-50%, -50%); }
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.6; }
                }
                @keyframes shimmer {
                    0% { background-position: -1000px 0; }
                    100% { background-position: 1000px 0; }
                }
            </style>
            
            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 12px; height: 12px; background: #00ffaa; border-radius: 50%; animation: pulse 1.5s infinite;"></div>
                    <h2 style="margin: 0; font-size: 20px; font-weight: 600; color: #00ffaa; text-shadow: 0 0 10px rgba(0, 255, 170, 0.5);">
                        🚀 Flux2 Training Progress
                    </h2>
                </div>
                <button id="flux-progress-minimize" style="background: rgba(255, 255, 255, 0.1); border: 1px solid #00ffaa; color: #00ffaa; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 600;">
                    MINIMIZE
                </button>
            </div>
            
            <!-- Package Installation Section -->
            <div id="package-install-section" style="display: none; margin-bottom: 25px;">
                <div style="font-size: 13px; color: #aaa; margin-bottom: 8px; font-weight: 500;">
                    📦 Installing Packages
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); border-radius: 12px; padding: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 15px; font-weight: 600; color: #00ffaa;" id="pkg-name">Initializing...</span>
                        <span style="font-size: 13px; color: #888;" id="pkg-count">0 / 9</span>
                    </div>
                    
                    <!-- Progress Bar -->
                    <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; overflow: hidden; height: 24px; margin-bottom: 10px;">
                        <div id="pkg-progress-bar" style="
                            background: linear-gradient(90deg, #00ffaa, #00ddff, #00aaff); 
                            height: 100%; 
                            width: 0%; 
                            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                            background-size: 200% 100%;
                            animation: shimmer 2s linear infinite;
                        "></div>
                    </div>
                    
                    <div style="font-size: 12px; color: #888;">
                        <span id="pkg-status">Preparing environment...</span>
                    </div>
                </div>
            </div>
            
            <!-- Training Progress Section -->
            <div id="training-section" style="display: none;">
                <div style="font-size: 13px; color: #aaa; margin-bottom: 8px; font-weight: 500;">
                    🎯 Training Progress
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); border-radius: 12px; padding: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 18px; font-weight: 700; color: #00ffaa;" id="train-percent">0%</span>
                        <span style="font-size: 13px; color: #888;" id="train-step">Step 0 / 0</span>
                    </div>
                    
                    <!-- Training Progress Bar -->
                    <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; overflow: hidden; height: 28px; margin-bottom: 12px;">
                        <div id="train-progress-bar" style="
                            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f); 
                            height: 100%; 
                            width: 0%; 
                            transition: width 0.5s ease;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: white;
                            font-weight: bold;
                            font-size: 12px;
                            text-shadow: 0 1px 3px rgba(0,0,0,0.8);
                        "></div>
                    </div>
                    
                    <!-- Stats Row -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 8px; border-radius: 6px;">
                            <div style="color: #888; margin-bottom: 3px;">Loss</div>
                            <div style="color: #00ffaa; font-weight: 600; font-size: 14px;" id="train-loss">-</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 8px; border-radius: 6px;">
                            <div style="color: #888; margin-bottom: 3px;">ETA</div>
                            <div style="color: #00ffaa; font-weight: 600; font-size: 14px;" id="train-eta">Calculating...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Status Footer -->
            <div id="status-footer" style="margin-top: 20px; padding: 12px; background: rgba(0, 255, 170, 0.1); border-radius: 8px; font-size: 12px; color: #00ffaa; text-align: center; font-weight: 500;">
                ⏳ Waiting for process to start...
            </div>
        `;
        
        document.body.appendChild(megaPanel);
        
        // Get DOM elements
        const pkgSection = megaPanel.querySelector("#package-install-section");
        const pkgName = megaPanel.querySelector("#pkg-name");
        const pkgCount = megaPanel.querySelector("#pkg-count");
        const pkgBar = megaPanel.querySelector("#pkg-progress-bar");
        const pkgStatus = megaPanel.querySelector("#pkg-status");
        
        const trainSection = megaPanel.querySelector("#training-section");
        const trainPercent = megaPanel.querySelector("#train-percent");
        const trainStep = megaPanel.querySelector("#train-step");
        const trainBar = megaPanel.querySelector("#train-progress-bar");
        const trainLoss = megaPanel.querySelector("#train-loss");
        const trainEta = megaPanel.querySelector("#train-eta");
        
        const statusFooter = megaPanel.querySelector("#status-footer");
        const minimizeBtn = megaPanel.querySelector("#flux-progress-minimize");
        
        // Minimize button handler
        minimizeBtn.addEventListener("click", () => {
            megaPanel.style.display = "none";
            console.log("[FLUX Progress] Panel minimized");
        });
        
        // Helper functions
        function showPanel() {
            megaPanel.style.display = "block";
        }
        
        function hidePanel() {
            megaPanel.style.display = "none";
        }
        
        function formatTime(seconds) {
            if (!seconds || seconds < 0) return "Unknown";
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            
            if (h > 0) return `${h}h ${m}m`;
            if (m > 0) return `${m}m ${s}s`;
            return `${s}s`;
        }
        
        // ======================================================================
        // EVENT LISTENER: Parse logs and update UI
        // ======================================================================
        
        api.addEventListener("flux_train_log", (event) => {
            const data = event.detail;
            if (!data || !data.line) return;
            
            const line = data.line;
            const lineLower = line.toLowerCase();
            
            // === PACKAGE INSTALLATION TRACKING ===
            if (lineLower.includes("[pkg]")) {
                showPanel();
                pkgSection.style.display = "block";
                
                // Extract package name and status
                const match = line.match(/\[PKG\]\s+([a-zA-Z0-9_-]+):\s+(\w+)/i);
                if (match) {
                    const pkg = match[1];
                    const status = match[2];
                    
                    if (status === "installing") {
                        state.packages.current++;
                        state.packages.currentName = pkg;
                        
                        pkgName.textContent = `Installing ${pkg}...`;
                        pkgCount.textContent = `${state.packages.current} / ${state.packages.total}`;
                        
                        const percent = (state.packages.current / state.packages.total) * 100;
                        pkgBar.style.width = percent + "%";
                        
                        pkgStatus.textContent = `Downloading and installing ${pkg} package`;
                        statusFooter.textContent = `⏳ Installing dependencies: ${percent.toFixed(0)}% complete`;
                    }
                    else if (status === "success") {
                        pkgStatus.textContent = `✓ ${pkg} installed successfully`;
                    }
                    else if (status === "failed" || status === "timeout") {
                        pkgStatus.textContent = `✗ ${pkg} installation ${status}`;
                        pkgName.style.color = "#ff6b6b";
                    }
                }
            }
            
            if (lineLower.includes("training packages ready") || lineLower.includes("setup complete")) {
                pkgSection.style.display = "none";
                statusFooter.textContent = "✅ Environment ready - Starting training";
            }
            
            // === TRAINING PROGRESS TRACKING ===
            if (lineLower.includes("training process started")) {
                showPanel();
                trainSection.style.display = "block";
                pkgSection.style.display = "none";
                state.training.startTime = Date.now();
                statusFooter.textContent = "🚀 Training in progress...";
            }
            
            // Extract max_train_steps from config
            const stepsMatch = line.match(/--max_train_steps['\s]+(\d+)/i) || 
                              line.match(/Configuration:.*?--max_train_steps\s+(\d+)/i);
            if (stepsMatch && state.training.totalSteps === 0) {
                state.training.totalSteps = parseInt(stepsMatch[1]);
                console.log(`[FLUX Progress] Detected total steps: ${state.training.totalSteps}`);
            }
            
            // Extract current step (various formats)
            const stepMatch = line.match(/step[s]?[:\s]+(\d+)[\s\/]+(\d+)/i) || 
                             line.match(/(\d+)\/(\d+)\s+step/i) ||
                             line.match(/epoch.*?(\d+)\/(\d+)/i);
            
            if (stepMatch) {
                const current = parseInt(stepMatch[1]);
                const total = parseInt(stepMatch[2]);
                
                if (total > 0) {
                    state.training.currentStep = current;
                    state.training.totalSteps = total;
                    
                    const percent = (current / total) * 100;
                    
                    trainPercent.textContent = percent.toFixed(1) + "%";
                    trainStep.textContent = `Step ${current} / ${total}`;
                    trainBar.style.width = percent + "%";
                    trainBar.textContent = `${current}/${total}`;
                    
                    // Calculate ETA
                    if (state.training.startTime && current > 0) {
                        const elapsed = (Date.now() - state.training.startTime) / 1000;
                        const timePerStep = elapsed / current;
                        const remaining = (total - current) * timePerStep;
                        trainEta.textContent = formatTime(remaining);
                    }
                    
                    statusFooter.textContent = `🎯 Training: ${percent.toFixed(1)}% complete`;
                }
            }
            
            // Extract loss
            const lossMatch = line.match(/loss[:\s=]+([0-9.]+)/i);
            if (lossMatch) {
                const loss = parseFloat(lossMatch[1]);
                state.training.lastLoss = loss;
                trainLoss.textContent = loss.toFixed(4);
            }
            
            // Training completion
            if (lineLower.includes("training process completed") || lineLower.includes("training finished")) {
                trainBar.style.width = "100%";
                trainPercent.textContent = "100%";
                statusFooter.textContent = "🎉 Training completed successfully!";
                statusFooter.style.background = "rgba(107, 207, 127, 0.3)";
                
                // Auto-hide after 8 seconds
                setTimeout(() => {
                    hidePanel();
                    // Reset state
                    state.training = { totalSteps: 0, currentStep: 0, startTime: null, lastLoss: null };
                    state.packages = { total: 9, current: 0, currentName: "", startTime: null };
                }, 8000);
            }
        });
        
        console.log("%c[FLUX Progress] ✓ Ready and Listening!", "color: lime; font-weight: bold; font-size: 14px;");
    }
});
