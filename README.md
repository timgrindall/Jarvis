# Jarvis

A local voice and text assistant that runs entirely on your home machine. No cloud APIs required.

## How It Works

Speak a query or type it, and Jarvis transcribes, thinks, and responds — either out loud or in text. Everything runs locally using Whisper for speech-to-text, Mistral 7B via Ollama for responses, and pyttsx3 for text-to-speech. Conversation history persists between sessions.

## Author Notes

I intentionally used a small model for running on small machines so it may feel limited at times, but it's still amazing software. This entire project was created using the free version of Claude.ai by Anthropic and started as simply as a hypothetical project which then became real when Claude asked, "Would you like me to create the files for this project?". A few iterations and user testing later, we got to version 5 as you see here.

The amazing this about this piece of software is that no personal info is shared across the web. It all runs locally on your machine. I wish software like this could become the future of AI instead of being stuck with a subscription to a cloud provider.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Mistral model pulled: `ollama pull mistral`
- Recommend 8GB RAM or more and a fast processor

## Installation

```bash
git clone https://github.com/yourname/jarvis
cd jarvis
pip install -r requirements.txt
```

## Usage

Open three terminals:

```bash
# Terminal 1
ollama serve

# Terminal 2
python jarvis_server.py

# Terminal 3
python jarvis_client.py
```

Once the client is running, press **Enter** to activate, then:

| Command | Action |
|---------|--------|
| `V` + Enter | Voice mode — hold SPACE to record, release to send |
| `T` + Enter | Text mode — type queries, blank line or ESC to exit |
| `S` + Enter | Standby |
| `quit` | Exit |

ESC exits voice or text mode and returns to standby.

## Configuration

Settings are at the top of `jarvis_server.py`:

```python
OLLAMA_MODEL = "mistral"   # swap to "llama2" for better quality
TOKEN_CAP    = 5000        # max conversation history tokens
```

Conversation history is saved to `jarvis_memory.json`. To clear it:

```bash
curl -X POST http://localhost:5000/memory/clear
```

## Stack

- [Whisper](https://github.com/openai/whisper) — speech to text
- [Ollama](https://ollama.ai) + Mistral 7B — local LLM
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — text to speech
- [Flask](https://flask.palletsprojects.com) — local server
- [sounddevice](https://python-sounddevice.readthedocs.io) — audio capture
