"""
Jarvis Earpiece Client
Voice mode → hold SPACE to record, release to send, plays audio response
Text mode  → type query, prints text response
ESC        → standby from any mode
pynput listener only active during voice recording session
"""

import sounddevice as sd
import numpy as np
import wave
import requests
import threading
import tempfile
import os
import sys
from pynput import keyboard

class JarvisEarpiece:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.recording = False
        self.audio_frames = []
        self.active = False
        self.space_released = threading.Event()

        self.CHANNELS = 1
        self.RATE = 16000

        self.TEMP_DIR = tempfile.gettempdir()
        self.INPUT_FILE = os.path.join(self.TEMP_DIR, "jarvis_input.wav")
        self.OUTPUT_FILE = os.path.join(self.TEMP_DIR, "jarvis_output.wav")

        self._print_banner()
        self._test_connection()

    def _print_banner(self):
        print("\n" + "="*50)
        print("JARVIS - STANDBY")
        print("="*50)
        print("\nPress Enter in this window to activate.")
        print("Jarvis will NOT listen until you do.\n")

    def _print_active(self):
        print("\n" + "="*50)
        print("JARVIS - ACTIVE")
        print("="*50)
        print("\nCommands:")
        print("  V + Enter     - Voice mode (hold SPACE to record)")
        print("  T + Enter     - Text query (prints response)")
        print("  S + Enter     - Standby")
        print("  ESC + Enter   - Standby (from any mode)")
        print("  quit + Enter  - Exit\n")

    def _go_standby(self):
        self.active = False
        print("\n[Standby] Press Enter to activate again.\n")

    def _test_connection(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code == 200:
                print("[OK] Server connected\n")
            else:
                print("[!!] Server returned error.")
        except requests.exceptions.ConnectionError:
            print("[!!] Cannot connect to server. Is jarvis_server.py running?\n")

    # ── Voice ────────────────────────────────────────────────────

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.audio_frames = []
        print("[Mic] Recording...")

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.RATE,
            channels=self.CHANNELS,
            dtype='int16',
            callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        if not self.recording:
            return None
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
            if not self.audio_frames:
                print("[!!] No audio recorded")
                return None
            audio_data = np.concatenate(self.audio_frames, axis=0)
            with wave.open(self.INPUT_FILE, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data.tobytes())
            print(f"[OK] Recording saved")
            return self.INPUT_FILE
        except Exception as e:
            print(f"[ERROR] Failed to save recording: {e}")
            return None

    def _cancel_recording(self):
        """Clean up any active recording stream"""
        self.recording = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass

    def do_voice_session(self):
        """
        Temporary pynput listener for this recording session only.
        Hold SPACE to record, release to send.
        ESC cancels and goes to standby.
        """
        print("Hold SPACE to record, release to send. ESC for standby.\n")
        self.space_released.clear()
        cancelled = threading.Event()

        def on_press(key):
            if key == keyboard.Key.space:
                if not self.recording:
                    self.start_recording()
            elif key == keyboard.Key.esc:
                cancelled.set()
                self.space_released.set()
                return False

        def on_release(key):
            if key == keyboard.Key.space:
                self.space_released.set()
                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            self.space_released.wait()
            listener.stop()

        if cancelled.is_set():
            self._cancel_recording()
            self._go_standby()
            return

        wav_file = self.stop_recording()
        if wav_file:
            thread = threading.Thread(
                target=self.process_voice_command,
                args=(wav_file,)
            )
            thread.daemon = True
            thread.start()

    def send_voice(self, wav_file):
        try:
            print("[>>] Sending to server...")
            with open(wav_file, 'rb') as f:
                response = requests.post(
                    f"{self.server_url}/process",
                    files={'audio': f},
                    timeout=120
                )
            if response.status_code == 200:
                return response.content
            else:
                print(f"[!!] Server error: {response.status_code}")
                return None
        except Exception as e:
            print(f"[!!] Error: {e}")
            return None

    def play_response(self, audio_bytes):
        if not audio_bytes:
            return
        try:
            with open(self.OUTPUT_FILE, 'wb') as f:
                f.write(audio_bytes)
            print("[>>] Playing response...\n")
            with wave.open(self.OUTPUT_FILE, 'rb') as wf:
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()),
                    dtype=np.int16
                )
                sd.play(audio_data, samplerate=wf.getframerate())
                sd.wait()
            print("[OK] Done\n")
        except Exception as e:
            print(f"[ERROR] Failed to play response: {e}\n")

    # ── Text ─────────────────────────────────────────────────────

    def do_text_session(self):
        """
        Reads a line of input. ESC (typed) or empty cancels to standby.
        Since input() is line-buffered we detect ESC as the '\x1b' character.
        """
        print("> ", end="", flush=True)
        try:
            text = input().strip()
        except EOFError:
            self._go_standby()
            return

        # ESC character or empty → standby
        if not text or text == '\x1b':
            self._go_standby()
            return

        thread = threading.Thread(
            target=self.process_text_command,
            args=(text,)
        )
        thread.daemon = True
        thread.start()

    def send_text(self, text):
        try:
            print("[>>] Sending to server...")
            response = requests.post(
                f"{self.server_url}/process_text",
                json={"text": text},
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"[!!] Server error: {response.status_code}")
                return None
        except Exception as e:
            print(f"[!!] Error: {e}")
            return None

    def print_response(self, text):
        if not text:
            return
        print("\n" + "-"*50)
        print(f"Jarvis: {text}")
        print("-"*50 + "\n")

    # ── Command handlers ─────────────────────────────────────────

    def process_voice_command(self, wav_file):
        audio = self.send_voice(wav_file)
        if audio:
            self.play_response(audio)

    def process_text_command(self, text):
        response = self.send_text(text)
        if response:
            self.print_response(response)

    # ── Main loop ────────────────────────────────────────────────

    def run(self):
        while True:
            cmd = input().strip().lower()

            # ESC character anywhere in main loop → standby
            if '\x1b' in cmd:
                if self.active:
                    self._go_standby()
                continue

            if not self.active:
                if cmd == "":
                    self.active = True
                    self._print_active()
                elif cmd == "quit":
                    print("Goodbye.")
                    sys.exit(0)
                continue

            if cmd == "quit":
                print("Goodbye.")
                sys.exit(0)

            elif cmd in ("s", ""):
                self._go_standby()

            elif cmd == "v":
                self.do_voice_session()

            elif cmd == "t":
                self.do_text_session()

            else:
                print("  V = voice  |  T = text  |  S/ESC = standby  |  quit = exit")


if __name__ == "__main__":
    earpiece = JarvisEarpiece(server_url="http://localhost:5000")
    try:
        earpiece.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
