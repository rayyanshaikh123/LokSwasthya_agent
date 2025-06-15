import os
import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, elevenlabs, openai, silero

# ── Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-intent")
load_dotenv()  # loads .env from current dir

# ── Language tables ──────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "hi": "Hindi",
}
DEEPGRAM_CODES = {c: c for c in LANGUAGE_NAMES}  # 1‑to‑1 mapping
GREETINGS = {
    "en": "Listening in English now.",
    "es": "¡Escuchando en español!",
    "fr": "J'écoute maintenant en français.",
    "de": "Ich höre jetzt Deutsch.",
    "it": "Ora ascolto in italiano.",
    "hi": "अब मैं हिंदी में सुन रहा हूँ।",
}

# ── Agent class ───────────────────────────────────────────────────────
class LanguageSwitcherAgent(Agent):
    """
    Replies in English via Ollama, but flips STT language when a helper
    GPT‑4o‑mini detects an intent like 'switch to Spanish'.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice health assistant named *LokSwasthya*, created by Rayyan Shaikh. "
"You help users with basic physical and mental health concerns in a conversational and caring tone. "
"You always respond in **simple English**, but you can understand questions in **multiple languages**. "
"Start by politely collecting the user's **name and phone number**. "
"Then ask about their symptoms, classify them as *mild*, *serious*, or *emergency*, and provide both **ayurvedic/natural** and **modern medical** suggestions where appropriate. "
"Show empathy, especially for mental health topics like depression, anxiety, or stress. "
"If symptoms are severe or urgent, gently advise the user to seek medical attention. "
"At the end, generate a structured **health report in proper list which can be easily readable** including the user's name, phone, symptoms, severity, and suggestions. "
"Use only English when replying, avoid emojis or special characters. Keep responses short, warm, and easy to follow."
"Avoid long responses. Keep each reply suitable for spoken interaction — clear, concise, and polite."

            ),
            stt=deepgram.STT(model="nova-3-general", language="multi"),
            tts=elevenlabs.TTS(
                model="eleven_turbo_v2_5",
                voice_id="iP95p4xoKVk53GoZ742B"
            ),
            llm=openai.LLM.with_ollama(
                model=os.getenv("OLLAMA_MODEL", "health-assistantv3"),
                base_url=os.getenv("OLLAMA_BASE_URL",
                                   "https://registered-toe-zip-scenarios.trycloudflare.com"),
            ),
            vad=silero.VAD.load(),
        )

        # tiny helper for intent detection (function‑calling not required)
        self.intent_llm = openai.LLM.with_ollama(
    model=os.getenv("OLLAMA_INTENT_MODEL", "llama3.2"),
    base_url=os.getenv("OLLAMA_BASE_URL", "https://registered-toe-zip-scenarios.trycloudflare.com"),
)

        self.current_lang = "en"

    # Initial greeting
    async def on_enter(self):
        await self.session.say(
    "Hi! I'm LokSwasthya. Please tell me your name and phone number to begin your health check."
)


    # Every final transcript arrives here
    async def on_transcription_complete(self, text: str):
        """Decide if the user asked to switch STT language."""
        target_code = await self._detect_language_intent(text)

        if target_code and target_code != self.current_lang:
            await self._apply_stt_language(target_code)
            return  # do NOT forward the phrase to Llama

        # Otherwise: normal pipeline → Llama3 → ElevenLabs TTS
        await self.session.generate_reply(text)

    # ---------- helper methods ---------------------------------------
    async def _detect_language_intent(self, text: str) -> str | None:
        """
        1. Fast keyword mapping (no LLM cost)
        2. If nothing matched, ask local Llama via intent_llm.
        """
        lowered = text.lower()

        # --- 1️⃣ keyword table (add more as needed) -----------------
        keyword_map = {
            "english": "en",
            "spanish": "es", "español": "es",
            "french": "fr",  "français": "fr",
            "german": "de",  "deutsch": "de",
            "italian": "it", "italiano": "it",
            "hindi": "hi",   "हिंदी": "hi",
        }
        for key, code in keyword_map.items():
            if key in lowered:
                logger.info(f"Keyword intent detected: {code}")
                return code

        # --- 2️⃣ fallback to Llama 3 prompt -------------------------
        prompt = (
            "Return ONLY one of these codes: en, es, fr, de, it, hi.\n"
            f'User: "{text.strip()}"\n'
            "If they want to switch the listening language, reply with that "
            "code. Otherwise reply with none."
        )
        try:
            resp = await self.intent_llm.complete(prompt, max_tokens=2)
            code = resp.strip().lower()
            if code in LANGUAGE_NAMES:
                logger.info(f"LLM intent detected language switch to {code}")
                return code
        except Exception as e:
            logger.warning(f"Intent LLM failed: {e}")

        return None

    async def _apply_stt_language(self, code: str):
        """Flip Deepgram STT and greet."""
        self.stt.update_options(language=DEEPGRAM_CODES[code])
        self.current_lang = code
        await self.session.say(GREETINGS[code])
        logger.info(f"STT language switched to {code}")

# ── Worker entrypoint ─────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info("Worker connected")

    session = AgentSession()
    await session.start(agent=LanguageSwitcherAgent(), room=ctx.room)
    logger.info("Agent session started")

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
