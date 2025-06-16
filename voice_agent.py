# agent.py
import os, logging
from dotenv import load_dotenv

from livekit.agents        import JobContext, WorkerOptions, cli
from livekit.agents.voice  import Agent, AgentSession
from livekit.plugins       import deepgram, elevenlabs, silero, groq
from openai                import AsyncOpenAI

# ── env / logging ──────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lokswasthya")

# ── Groq LLM (single backend) ─────────────────────────────
groq_llm = groq.LLM(
    model   = os.getenv("GROQ_MODEL", "llama3-8b-8192"),
    api_key = os.getenv("GROQ_API_KEY"),
)

# ── language tables ───────────────────────────────────────

# ── Agent ─────────────────────────────────────────────────
class LanguageSwitcherAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly voice health assistant named LokSwasthya, "
                "created by Rayyan Shaikh. Respond in simple English, understand "
                "multiple languages, collect name & phone, classify symptoms, "
                "give ayurvedic and modern suggestions, be concise, and produce "
                "a readable health report."
            ),
            stt=deepgram.STT(model="nova-3-general", language="multi"),
            tts=elevenlabs.TTS(
                model="eleven_turbo_v2_5",
                voice_id="cgSgspJ2msm6clMCkdW9",
            ),
            llm=groq_llm,          # ← single LLM backend
            vad=silero.VAD.load(),
        )

        # tiny intent detector (local / Ollama not needed)
        self.intent_llm = groq_llm
        self.current_lang = "en"

    # greeting
    async def on_enter(self):
        await self.session.say(
            "Hi! I'm LokSwasthya. Please tell me your name and phone number to begin your health check."
        )

    async def on_transcription_complete(self, text: str):
        code = await self._detect_language_intent(text)
        if code and code != self.current_lang:
            await self._apply_stt_language(code)
            return

        reply = await self.llm.complete(text)
        await self.session.say(reply)
        await self.session.send_text(reply)

    # intent detector
    
# ── worker entrypoint ─────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()
    await session.start(agent=LanguageSwitcherAgent(), room=ctx.room)

# ── main ──────────────────────────────────────────────────
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
