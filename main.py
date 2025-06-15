"""
main.py – Render Web Service entrypoint
• Serves /  →  health‑check JSON
• Spawns `voice_agent.py` as a child process on startup
Run locally:
    uvicorn main:app --reload        (agent will also start)
"""
import os, subprocess, threading, time, logging, signal
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lokswasthya-main")

AGENT_CMD = ["python", "voice_agent.py", "dev"]
 # <- match your filename

def agent_supervisor():
    """
    Launch the LiveKit agent as a child process.
    If it ever exits, restart after 5 seconds.
    """
    while True:
        logger.info("Starting LiveKit agent subprocess …")
        proc = subprocess.Popen(AGENT_CMD, env=os.environ.copy())
        exit_code = proc.wait()
        logger.warning("Agent exited (code=%s). Restarting in 5 s …", exit_code)
        time.sleep(5)

# ── FastAPI app (Render listens on this) ──────────────────────────────
app = FastAPI()

@app.get("/")
def health():
    return {"status": "LokSwasthya web service running."}

@app.on_event("startup")
def launch_agent():
    # Start supervisor in a daemon thread so it doesn't block FastAPI
    threading.Thread(target=agent_supervisor, daemon=True).start()
    logger.info("Agent supervisor thread launched")
