/**
 * Progress Tracker Extension for Flux2 Training Monitor
 * Adds real-time progress bars for training steps, epochs, and package installation
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("%c[Progress Tracker] Loading...", "color: magenta; font-weight: bold;");

// Progress state
const progressState = {
    training: {
        current: 0,
        total: 0,
        percent: 0,
        active: false
    },
    packages: {
        current: 0,
        total: 0,
        currentPackage: "",
        active: false
    }
};

app.registerExtension({
    name: "Flux2.LoRA.ProgressTracker",
    
    async setup() {
        console.log("[Progress Tracker] Initializing");
        
        // Create progress panel
        const progressPanel = document.createElement("div");
        progressPanel.id = "flux-progress-panel";
        
        Object.assign(progressPanel.style, {
            position: "fixed",
            top: "10px",
            right: "10px",
            width: "400px",
            background: "rgba(20, 20, 30, 0.95)",
            color: "#fff",
            fontFamily: "Arial, sans-serif",
            fontSize: "13px",
            padding: "15px",
            borderRadius: "8px",
            border: "2px solid #4CAF50",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
            zIndex: "10001",
            display: "none",
            backdropFilter: "blur(5px)"
        });
        
        progressPanel.innerHTML = `
            <div style="margin-bottom: 15px; font-weight: bold; color: #4CAF50; font-size: 15px;">
                ⏱️ Training Progress
            </div>
            
            <!-- Training Progress -->
            <div id="training-progress-section" style="display: none;">
                <div style="margin-bottom: 5px; font-size: 12px; color: #aaa;">
                    Training Step: <span id="training-step-text">0 / 0</span>
                </div>
                <div style="background: #2a2a3a; border-radius: 10px; overflow: hidden; height: 20px; margin-bottom: 10px;">
                    <div id="training-progress-bar" style="background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
                </div>
                <div style="font-size: 11px; color: #888;">
                    <span id="training-percent">0%</span> complete
                </div>
            </div>
            
            <!-- Package Installation Progress -->
            <div id="package-progress-section" style="display: none; margin-top: 15px;">
                <div style="margin-bottom: 5px; font-size: 12px; color: #aaa;">
                    Installing: <span id="package-name" style="color: #4CAF50;">packages</span>
                </div>
                <div style="background: #2a2a3a; border-radius: 10px; overflow: hidden; height: 16px; margin-bottom: 5px;">
                    <div id="package-progress-bar" style="background: linear-gradient(90deg, #2196F3, #00BCD4); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
                </div>
                <div style="font-size: 11px; color: #888;">
                    <span id="package-count">0 / 0</span> packages
                </div>
            </div>
            
            <!-- Status Messages -->
            <div id="progress-status" style="margin-top: 15px; padding: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; font-size: 11px; color: #ccc; max-height: 60px; overflow-y: auto;">
                Waiting for training to start...
            </div>
        `;
        
        document.body.appendChild(progressPanel);
        
        // Get DOM elements
        const trainingSection = progressPanel.querySelector("#training-progress-section");
        const trainingBar = progressPanel.querySelector("#training-progress-bar");
        const trainingStepText = progressPanel.querySelector("#training-step-text");
        const trainingPercent = progressPanel.querySelector("#training-percent");
        
        const packageSection = progressPanel.querySelector("#package-progress-section");
        const packageBar = progressPanel.querySelector("#package-progress-bar");
        const packageName = progressPanel.querySelector("#package-name");
        const packageCount = progressPanel.querySelector("#package-count");
        
        const statusDiv = progressPanel.querySelector("#progress-status");
        
        // Helper: Update status message
        function updateStatus(message) {
            const timestamp = new Date().toLocaleTimeString();
            statusDiv.textContent = `[${timestamp}] ${message}`;
        }
        
        // Helper: Show/hide panel
        function showPanel() {
            progressPanel.style.display = "block";
        }
        
        function hidePanel() {
            progressPanel.style.display = "none";
        }
        
        // Listen for training log events and extract progress
        api.addEventListener("flux_train_log", (event) => {
            const data = event.detail;
            if (!data || !data.line) return;
            
            const line = data.line.toLowerCase();
            
            // === PACKAGE INSTALLATION DETECTION ===
            if (line.includes("installing") && (line.includes("packages") || line.includes("pytorch"))) {
                showPanel();
                packageSection.style.display = "block";
                updateStatus("Installing training packages...");
            }
            
            if (line.includes("installing") && line.includes("==")) {
                // Extract package name (e.g., "Installing transformers==4.36.2...")
                const match = data.line.match(/Installing\s+([a-zA-Z0-9_-]+)==/);
                if (match) {
                    packageName.textContent = match[1];
                    progressState.packages.currentPackage = match[1];
                    progressState.packages.current++;
                    
                    const percent = progressState.packages.total > 0 
                        ? (progressState.packages.current / progressState.packages.total) * 100
                        : 0;
                    
                    packageBar.style.width = percent + "%";
                    packageCount.textContent = `${progressState.packages.current} / ${progressState.packages.total}`;
                }
            }
            
            if (line.includes("training packages ready") || line.includes("setup complete")) {
                packageSection.style.display = "none";
                updateStatus("Package installation complete ✓");
            }
            
            // === TRAINING PROGRESS DETECTION ===
            if (line.includes("--- training process started ---")) {
                showPanel();
                trainingSection.style.display = "block";
                updateStatus("Training started");
            }
            
            // Extract step numbers (e.g., "step 50/1200", "steps: 50/1200")
            const stepMatch = data.line.match(/step[s]?[:\s]+(\d+)\s*[\/\|]\s*(\d+)/i);
            if (stepMatch) {
                const current = parseInt(stepMatch[1]);
                const total = parseInt(stepMatch[2]);
                
                progressState.training.current = current;
                progressState.training.total = total;
                progressState.training.percent = (current / total) * 100;
                progressState.training.active = true;
                
                trainingStepText.textContent = `${current} / ${total}`;
                trainingPercent.textContent = progressState.training.percent.toFixed(1) + "%";
                trainingBar.style.width = progressState.training.percent + "%";
                
                updateStatus(`Training step ${current}/${total}`);
            }
            
            // Extract loss values
            const lossMatch = data.line.match(/loss[:\s]+([0-9.]+)/i);
            if (lossMatch) {
                const loss = parseFloat(lossMatch[1]);
                updateStatus(`Loss: ${loss.toFixed(4)}`);
            }
            
            // Training completion
            if (line.includes("training process completed") || line.includes("training finished")) {
                trainingBar.style.width = "100%";
                trainingPercent.textContent = "100%";
                updateStatus("Training completed ✓");
                
                // Auto-hide after 5 seconds
                setTimeout(() => {
                    hidePanel();
                    // Reset state
                    progressState.training.active = false;
                    progressState.training.current = 0;
                    progressState.training.total = 0;
                }, 5000);
            }
        });
        
        // Initialize package count (estimate based on requirements)
        progressState.packages.total = 9; // torch, transformers, diffusers, accelerate, etc.
        
        console.log("%c[Progress Tracker] Ready!", "color: magenta; font-weight: bold;");
    }
});
