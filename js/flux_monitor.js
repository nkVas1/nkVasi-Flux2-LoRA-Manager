/**
 * FLUX.2 LoRA Training Monitor Extension (v2.0)
 * 
 * Displays real-time training logs in a floating panel within ComfyUI browser interface.
 * Receives log data via WebSocket from Python backend (TrainingProcessManager).
 * 
 * Key improvements:
 * - Console logging for debugging
 * - Unbuffered output handling
 * - Better error detection
 * - Robust registration
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("%c[FLUX Monitor] Loading Extension...", "color: cyan; font-weight: bold; font-size: 14px;");

app.registerExtension({
    name: "Flux2.LoRA.Monitor",
    
    async setup() {
        console.log("%c[FLUX Monitor] Setup Started", "color: cyan; font-weight: bold;");
        
        // Remove old panel if exists (for script reloads)
        const oldPanel = document.getElementById("flux-monitor-panel");
        if (oldPanel) oldPanel.remove();
        
        const logPanel = document.createElement("div");
        logPanel.id = "flux-monitor-panel";
        
        // Styling for the log panel - positioned at bottom right, always on top
        Object.assign(logPanel.style, {
            position: "fixed",
            bottom: "10px",
            right: "10px",
            width: "60%",
            maxWidth: "900px",
            height: "320px",
            background: "rgba(10, 10, 20, 0.97)",
            color: "#00ff00",
            fontFamily: "Consolas, 'Courier New', monospace",
            fontSize: "11px",
            lineHeight: "1.3",
            overflowY: "auto",
            overflowX: "hidden",
            zIndex: "10000",
            padding: "10px",
            borderRadius: "4px",
            border: "2px solid #00ff00",
            boxShadow: "0 0 20px rgba(0, 255, 0, 0.3), inset 0 0 10px rgba(0, 255, 0, 0.1)",
            display: "none",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            scrollBehavior: "smooth",
            pointerEvents: "auto"
        });
        
        document.body.appendChild(logPanel);

        // =====================================================================
        // HEADER WITH CLOSE BUTTON
        // =====================================================================
        
        const header = document.createElement("div");
        Object.assign(header.style, {
            position: "sticky",
            top: "0",
            background: "rgba(0, 100, 0, 0.5)",
            borderBottom: "1px solid #00ff00",
            padding: "5px 8px",
            marginBottom: "5px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            cursor: "pointer",
            userSelect: "none"
        });
        
        header.innerHTML = `
            <span><strong style="color: #00ff00;">🚀 FLUX.2 Training Monitor</strong></span>
            <span style="font-size: 10px; color: #888; cursor: pointer;">[CLOSE]</span>
        `;
        
        const closeBtn = header.querySelector("span:last-child");
        closeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            logPanel.style.display = "none";
            console.log("[FLUX Monitor] Panel closed by user");
        });
        
        logPanel.appendChild(header);

        // =====================================================================
        // LOG CONTENT AREA
        // =====================================================================
        
        const contentDiv = document.createElement("div");
        contentDiv.style.maxHeight = "280px";
        contentDiv.style.overflowY = "auto";
        contentDiv.style.fontFamily = "Consolas, monospace";
        logPanel.appendChild(contentDiv);

        // =====================================================================
        // EVENT LISTENERS
        // =====================================================================
        
        // Listen for log events from Python backend
        api.addEventListener("flux_train_log", (event) => {
            const data = event.detail;
            
            if (!data || !data.line) {
                console.warn("[FLUX Monitor] Received empty event");
                return;
            }

            // Show panel when first log arrives
            if (logPanel.style.display === "none") {
                logPanel.style.display = "block";
                console.log("[FLUX Monitor] Log panel shown");
            }

            // Create log line
            const line = document.createElement("div");
            line.textContent = `> ${data.line}`;
            line.style.borderBottom = "1px solid #1a1a2a";
            line.style.paddingBottom = "2px";
            line.style.marginBottom = "2px";
            
            // Color-code based on content
            const textLower = data.line.toLowerCase();
            
            if (textLower.includes("error") || textLower.includes("exception") || textLower.includes("traceback")) {
                line.style.color = "#ff6b6b";
                line.style.fontWeight = "bold";
            } else if (textLower.includes("warning")) {
                line.style.color = "#ffff00";
            } else if (textLower.includes("success") || textLower.includes("completed") || textLower.includes("finished")) {
                line.style.color = "#00ff00";
                line.style.fontWeight = "bold";
            } else if (textLower.includes("step") || textLower.includes("epoch") || textLower.includes("loss") || textLower.includes("%")) {
                line.style.color = "#00ccff";
            } else if (textLower.includes("loading") || textLower.includes("preparing") || textLower.includes("initializing")) {
                line.style.color = "#ffaa00";
            } else if (textLower.includes("started") || textLower.includes("begin")) {
                line.style.color = "#88ff88";
            }
            
            contentDiv.appendChild(line);
            
            // Limit buffer (prevent browser freeze)
            if (contentDiv.childElementCount > 1000) {
                contentDiv.removeChild(contentDiv.firstChild);
            }

            // Auto-scroll to bottom
            contentDiv.scrollTop = contentDiv.scrollHeight;
            
            // Log to browser console
            console.log("[FLUX-TRAIN]", data.line);
        });
        
        console.log("%c[FLUX Monitor] Ready and Listening!", "color: lime; font-weight: bold; font-size: 12px;");
    }
});
