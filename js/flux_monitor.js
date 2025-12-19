/**
 * FLUX.2 LoRA Training Monitor Extension
 * 
 * Displays real-time training logs in a floating panel within ComfyUI browser interface.
 * Receives log data via WebSocket from Python backend (TrainingProcessManager).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Flux2.LoRA.Monitor",
    
    async setup() {
        // =====================================================================
        // CREATE FLOATING LOG PANEL
        // =====================================================================
        
        const logPanel = document.createElement("div");
        
        // Styling for the log panel
        Object.assign(logPanel.style, {
            position: "fixed",
            bottom: "10px",
            right: "10px",
            width: "600px",
            height: "350px",
            background: "rgba(0, 0, 0, 0.95)",
            color: "#00ff00",
            fontFamily: "'Courier New', monospace",
            fontSize: "12px",
            lineHeight: "1.4",
            overflowY: "auto",
            overflowX: "hidden",
            zIndex: "9999",
            padding: "12px",
            borderRadius: "6px",
            border: "2px solid #00ff00",
            boxShadow: "0 0 20px rgba(0, 255, 0, 0.3)",
            display: "none", // Hidden by default
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            scrollBehavior: "smooth"
        });
        
        document.body.appendChild(logPanel);

        // =====================================================================
        // HEADER/TITLE BAR
        // =====================================================================
        
        const titleBar = document.createElement("div");
        Object.assign(titleBar.style, {
            position: "absolute",
            top: "0",
            left: "0",
            right: "0",
            height: "30px",
            background: "rgba(0, 150, 0, 0.3)",
            borderBottom: "1px solid #00ff00",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingLeft: "10px",
            paddingRight: "10px",
            cursor: "move",
            userSelect: "none"
        });
        
        titleBar.innerHTML = `
            <span style="font-weight: bold; color: #00ff00;">🚀 FLUX.2 Training Monitor</span>
            <span style="font-size: 10px; color: #888;">Click to close</span>
        `;
        
        logPanel.appendChild(titleBar);

        // =====================================================================
        // LOG CONTENT AREA
        // =====================================================================
        
        const logContent = document.createElement("div");
        Object.assign(logContent.style, {
            position: "absolute",
            top: "35px",
            left: "0",
            right: "0",
            bottom: "0",
            overflowY: "auto",
            overflowX: "hidden",
            padding: "10px",
            fontSize: "11px"
        });
        
        logPanel.appendChild(logContent);

        // =====================================================================
        // STATUS INDICATOR
        // =====================================================================
        
        let trainingActive = false;
        const statusDot = document.createElement("span");
        statusDot.style.display = "inline-block";
        statusDot.style.width = "8px";
        statusDot.style.height = "8px";
        statusDot.style.borderRadius = "50%";
        statusDot.style.marginRight = "5px";
        statusDot.style.background = "#888";
        statusDot.style.animation = "none";
        
        // Add blinking animation CSS
        const style = document.createElement("style");
        style.innerHTML = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            .flux-training-active {
                animation: pulse 1s infinite;
                background: #00ff00 !important;
            }
        `;
        document.head.appendChild(style);

        // =====================================================================
        // EVENT LISTENERS
        // =====================================================================
        
        // Listen for log events from Python backend
        api.addEventListener("flux_train_log", (event) => {
            const data = event.detail;
            
            if (!data) return;

            // Show panel when training starts
            if (logPanel.style.display === "none") {
                logPanel.style.display = "block";
                logContent.innerHTML = "";
                trainingActive = true;
                statusDot.className = "flux-training-active";
                titleBar.innerHTML = `
                    <span>${statusDot.outerHTML}<span style="font-weight: bold; color: #00ff00;">🚀 FLUX.2 Training Monitor [RUNNING]</span></span>
                    <span style="font-size: 10px; color: #888;">Click to close</span>
                `;
            }

            // Add log line
            const logLine = document.createElement("div");
            logLine.textContent = data.line || String(data);
            
            // Color-code based on content
            if (data.line) {
                const text = data.line.toLowerCase();
                if (text.includes("error") || text.includes("exception") || text.includes("cuda")) {
                    logLine.style.color = "#ff6b6b";
                    logLine.style.fontWeight = "bold";
                } else if (text.includes("warning")) {
                    logLine.style.color = "#ffff00";
                } else if (text.includes("finished") || text.includes("success")) {
                    logLine.style.color = "#00ff00";
                    logLine.style.fontWeight = "bold";
                } else if (text.includes("step") || text.includes("epoch") || text.includes("loss")) {
                    logLine.style.color = "#00ccff";
                } else if (text.includes("loading") || text.includes("preparing")) {
                    logLine.style.color = "#ffaa00";
                }
            }
            
            logContent.appendChild(logLine);
            
            // Auto-scroll to bottom
            logContent.scrollTop = logContent.scrollHeight;

            // Mark completion
            if (data.line && (data.line.includes("Training Finished") || data.line.includes("Error occurred"))) {
                trainingActive = false;
                statusDot.className = "";
                statusDot.style.background = data.line.includes("Error") ? "#ff0000" : "#00ff00";
                const finalLine = document.createElement("div");
                finalLine.textContent = data.line.includes("Error") ? "❌ TRAINING FAILED" : "✅ TRAINING COMPLETED";
                finalLine.style.color = data.line.includes("Error") ? "#ff0000" : "#00ff00";
                finalLine.style.fontWeight = "bold";
                finalLine.style.marginTop = "10px";
                finalLine.style.padding = "5px";
                finalLine.style.borderTop = "1px solid #555";
                logContent.appendChild(finalLine);
            }
        });

        // Click to close panel
        titleBar.addEventListener("click", () => {
            logPanel.style.display = "none";
            trainingActive = false;
            statusDot.className = "";
        });

        // =====================================================================
        // DRAGGING FUNCTIONALITY
        // =====================================================================
        
        let isDragging = false;
        let dragOffsetX = 0;
        let dragOffsetY = 0;

        titleBar.addEventListener("mousedown", (e) => {
            isDragging = true;
            dragOffsetX = e.clientX - logPanel.offsetLeft;
            dragOffsetY = e.clientY - logPanel.offsetTop;
        });

        document.addEventListener("mousemove", (e) => {
            if (isDragging) {
                logPanel.style.left = (e.clientX - dragOffsetX) + "px";
                logPanel.style.top = (e.clientY - dragOffsetY) + "px";
                logPanel.style.right = "auto";
                logPanel.style.bottom = "auto";
            }
        });

        document.addEventListener("mouseup", () => {
            isDragging = false;
        });

        // =====================================================================
        // DEBUG: Log when extension is loaded
        // =====================================================================
        
        console.log("[FLUX Monitor] Extension loaded successfully");
    }
});
