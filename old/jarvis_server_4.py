"""
Jarvis Home Server (Local LLM + Persistent Memory)
Voice requests → audio response
Text requests  → text response
"""

from flask import Flask, request, send_file, jsonify
import whisper
import requests
import json
from io import BytesIO
import os
import tempfile
from datetime import datetime

TEMP_DIR = tempfile.gettempdir()
AUDIO_INPUT = os.path.join(TEMP_DIR, "jarvis_input.wav")
AUDIO_OUTPUT = os.path.join(TEMP_DIR, "jarvis_output.wav")

app = Flask(__name__)

# Initialize Whisper
whisper_model = whisper.load_model("base")

# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # Default: "George" (deep, clear)
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434"
MEMORY_FILE = "jarvis_memory.json"
TOKEN_CAP = 1000

SYSTEM_PROMPT_VOICE = (
    "You are Jarvis, a helpful voice assistant. "
    "Give clear, natural responses of around 3-4 sentences suitable for speaking aloud. "
    "Be conversational and informative without being too brief or too long."
)

SYSTEM_PROMPT_TEXT = (
    "You are Jarvis, a helpful assistant. "
    "Give well-structured responses of around 3-4 sentences. "
    "Be conversational, clear, and informative."
)

print("[SERVER] Jarvis initialized. Waiting for commands...")


# ── Memory ───────────────────────────────────────────────────────

def load_history():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                history = json.load(f)
            print(f"[MEMORY] Loaded {len(history)} messages from disk")
            return history
        except Exception as e:
            print(f"[MEMORY] Failed to load history: {e}")
    return []


def save_history(history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[MEMORY] Failed to save history: {e}")


def estimate_tokens(history):
    return sum(len(m["content"]) for m in history) // 4


def trim_history(history):
    while estimate_tokens(history) > TOKEN_CAP and len(history) > 2:
        history.pop(0)
        if history and history[0]["role"] == "assistant":
            history.pop(0)
    print(f"[MEMORY] {len(history)} messages (~{estimate_tokens(history)} tokens)")
    return history


conversation_history = load_history()


# ── Core pipeline ────────────────────────────────────────────────

def transcribe_audio(audio_bytes):
    try:
        with open(AUDIO_INPUT, "wb") as f:
            f.write(audio_bytes)
        result = whisper_model.transcribe(AUDIO_INPUT)
        text = result["text"].strip()
        print(f"[STT] Transcribed: {text}")
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None


def web_search(query, num_results=5):
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "engine": "google"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=5)
        results = response.json()
        search_results = []
        if "organic_results" in results:
            for r in results["organic_results"][:3]:
                search_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                })
        print(f"[SEARCH] Found {len(search_results)} results")
        return search_results
    except Exception as e:
        print(f"[ERROR] Web search failed: {e}")
        return []


def generate_response(user_query, search_results, system_prompt):
    """Send conversation history + new query to Ollama"""
    global conversation_history

    if search_results:
        results_text = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])
        content = f"{user_query}\n\nSearch results:\n{results_text}"
    else:
        content = user_query

    conversation_history.append({"role": "user", "content": content})
    conversation_history = trim_history(conversation_history)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + conversation_history,
                "stream": False,
            },
            timeout=180
        )

        if response.status_code == 200:
            answer = response.json()["message"]["content"].strip()
            print(f"[RESPONSE] {answer}")
            conversation_history.append({"role": "assistant", "content": answer})
            save_history(conversation_history)
            return answer
        else:
            print(f"[ERROR] Ollama error: {response.status_code}")
            return "I'm having trouble processing that."

    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama.")
        return "Ollama server is not running. Please start it."
    except Exception as e:
        print(f"[ERROR] Response generation failed: {e}")
        return "I'm sorry, I had trouble processing that."


def text_to_speech(text):
    """Use ElevenLabs if key is set, otherwise fall back to pyttsx3"""
    if ELEVENLABS_KEY:
        return tts_elevenlabs(text)
    else:
        return tts_local(text)


def tts_elevenlabs(text):
    """High quality TTS via ElevenLabs API"""
    try:
        # Request WAV format so client can play it directly
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=pcm_16000"
        headers = {
            "xi-api-key": ELEVENLABS_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            # Wrap raw PCM in WAV header so client can play it
            import wave, struct
            pcm_data = response.content
            import io
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit
                wf.setframerate(16000)
                wf.writeframes(pcm_data)
            audio_bytes = wav_buffer.getvalue()
            print(f"[TTS] ElevenLabs generated audio ({len(audio_bytes)} bytes)")
            return audio_bytes
        else:
            print(f"[TTS] ElevenLabs error {response.status_code}: {response.text}")
            print("[TTS] Falling back to local TTS")
            return tts_local(text)

    except Exception as e:
        print(f"[ERROR] ElevenLabs TTS failed: {e}")
        print("[TTS] Falling back to local TTS")
        return tts_local(text)


def tts_local(text):
    """Fallback TTS using pyttsx3 (no API key needed)"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(text, AUDIO_OUTPUT)
        engine.runAndWait()
        with open(AUDIO_OUTPUT, "rb") as f:
            audio_bytes = f.read()
        print(f"[TTS] Local TTS generated audio ({len(audio_bytes)} bytes)")
        return audio_bytes
    except Exception as e:
        print(f"[ERROR] Local TTS failed: {e}")
        return None


# ── Routes ───────────────────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process_voice():
    """Voice in → audio out"""
    try:
        print("\n" + "="*50)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Voice command...")

        if "audio" not in request.files:
            return {"error": "No audio file provided"}, 400

        audio_bytes = request.files["audio"].read()
        user_query = transcribe_audio(audio_bytes)
        if not user_query:
            return {"error": "Transcription failed"}, 500

        search_results = web_search(user_query) if SERPAPI_KEY else []
        if not SERPAPI_KEY:
            print("[SEARCH] Skipped (no API key)")

        response_text = generate_response(user_query, search_results, SYSTEM_PROMPT_VOICE)
        response_audio = text_to_speech(response_text)

        if response_audio:
            return send_file(BytesIO(response_audio), mimetype="audio/wav")
        else:
            return {"error": "TTS generation failed"}, 500

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return {"error": str(e)}, 500


@app.route("/process_text", methods=["POST"])
def process_text():
    """Text in → text out"""
    try:
        print("\n" + "="*50)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Text query...")

        data = request.get_json()
        if not data or "text" not in data:
            return {"error": "No text provided"}, 400

        user_query = data["text"].strip()
        if not user_query:
            return {"error": "Empty query"}, 400

        print(f"[TEXT] Query: {user_query}")

        search_results = web_search(user_query) if SERPAPI_KEY else []
        if not SERPAPI_KEY:
            print("[SEARCH] Skipped (no API key)")

        response_text = generate_response(user_query, search_results, SYSTEM_PROMPT_TEXT)

        # Return plain text, not audio
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"[ERROR] Text processing failed: {e}")
        return {"error": str(e)}, 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "online", "messages": len(conversation_history), "timestamp": datetime.now().isoformat()})


@app.route("/memory/clear", methods=["POST"])
def clear_memory():
    global conversation_history
    conversation_history = []
    save_history(conversation_history)
    print("[MEMORY] Cleared")
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("JARVIS HOME SERVER (LOCAL LLM + MEMORY)")
    print("="*50)
    print(f"Model:     {OLLAMA_MODEL}")
    print(f"Memory:    {MEMORY_FILE}")
    print(f"Token cap: {TOKEN_CAP}")
    print(f"TTS:       {'ElevenLabs' if ELEVENLABS_KEY else 'pyttsx3 (local)'}")
    print(f"Search:    {'SerpAPI' if SERPAPI_KEY else 'Disabled'}")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
