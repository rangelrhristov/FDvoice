import ctypes
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tempfile
import wave
from pathlib import Path

import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from pycaw.pycaw import AudioUtilities
except Exception:
    AudioUtilities = None

try:
    import comtypes
except Exception:
    comtypes = None


APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "FDvoice"
CONFIG_PATH = APP_DIR / "config.json"
LOG_PATH = APP_DIR / "engine.log"
HISTORY_PATH = APP_DIR / "history.jsonl"

SAMPLE_RATE = 16000
CHANNELS = 1
CONFIG_VERSION = 14
ERROR_ALREADY_EXISTS = 183
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_V = 0x56
VK_Z = 0x5A
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
ENGINE_MUTEX_HANDLE = None


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_uint),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("union", INPUT_UNION)]


DEFAULT_PHRASE_CORRECTIONS = {
    "fd voice": "FDvoice",
    "f d voice": "FDvoice",
    "eft voice": "FDvoice",
}

DEFAULT_WHISPER_HOTWORDS = "FDvoice Codex Groq Tauri Windows YouTube Music IBKR Binance Nvidia Parakeet"

DEFAULT_TEXT_CLEANUP_PROMPT = """Clean this dictated text for direct insertion.

Rules:
- Remove filler words like um, uh, ah, hmm.
- Remove false starts and repeated words.
- Keep the wording casual.
- Do not make it formal.
- Use minimal punctuation.
- Avoid commas unless needed for meaning.
- Do not add markdown unless I clearly dictated markdown.
- Preserve coding terms, package names, filenames, commands, acronyms, and product names.
- Return only the cleaned text.

Text:
{{transcript}}"""

MIGRATED_DEFAULT_KEYS = [
    "ASRBackend",
    "DictationMode",
    "PauseMediaWhileDictating",
    "CohereModel",
    "CoherePunctuation",
    "CohereMaxNewTokens",
    "CohereFallbackBackend",
    "WhisperModel",
    "WhisperDevice",
    "WhisperComputeType",
    "FallbackWhisperModel",
    "FallbackWhisperDevice",
    "FallbackWhisperComputeType",
    "WhisperHotwords",
    "WhisperNoSpeechThreshold",
    "WhisperLogProbThreshold",
    "WhisperCompressionRatioThreshold",
    "WhisperHallucinationSilenceThreshold",
    "WhisperVadMinSilenceMs",
    "WhisperVadSpeechPadMs",
    "ParakeetLang",
    "ParakeetDevice",
    "ParakeetDType",
    "ParakeetWarmupSeconds",
    "GroqModel",
    "GroqAccuracyModel",
    "GroqFallbackModel",
    "GroqApiKeyEnv",
    "GroqPrompt",
    "TextCleanupEnabled",
    "TextCleanupModel",
    "TextCleanupMaxTokens",
    "TextCleanupPrompt",
    "WindowsRecognitionCulture",
    "WindowsRecognitionTimeoutSeconds",
    "PreloadModelOnEngineStart",
    "WarmupModelOnEngineStart",
    "StreamingTranscription",
    "StreamChunkSeconds",
    "MinimumAudioSeconds",
    "MinimumAudioRms",
    "PostProcessPunctuation",
    "SmartFormattingEnabled",
    "PhraseCorrections",
]

DEFAULT_CONFIG = {
    "ConfigVersion": CONFIG_VERSION,
    "ASRBackend": "cohere",
    "DictationMode": "default",
    "HotkeyMode": "Ctrl+Win",
    "PauseMediaWhileDictating": True,
    "MusicProcessNames": [
        "chrome",
        "msedge",
        "firefox",
        "brave",
        "opera",
        "spotify",
        "YouTube Music",
        "youtube music",
    ],
    "CohereModel": "CohereLabs/cohere-transcribe-03-2026",
    "CoherePunctuation": True,
    "CohereMaxNewTokens": 256,
    "CohereFallbackBackend": "groq",
    "WhisperModel": "distil-whisper/distil-large-v3.5-ct2",
    "WhisperDevice": "cuda",
    "WhisperComputeType": "float16",
    "FallbackWhisperModel": "small.en",
    "FallbackWhisperDevice": "cpu",
    "FallbackWhisperComputeType": "int8",
    "WhisperHotwords": DEFAULT_WHISPER_HOTWORDS,
    "WhisperNoSpeechThreshold": 0.6,
    "WhisperLogProbThreshold": -1.0,
    "WhisperCompressionRatioThreshold": 2.4,
    "WhisperHallucinationSilenceThreshold": 1.0,
    "WhisperVadMinSilenceMs": 500,
    "WhisperVadSpeechPadMs": 250,
    "ParakeetLang": "EN",
    "ParakeetDevice": "cuda",
    "ParakeetDType": "float16",
    "ParakeetWarmupSeconds": [1, 3, 6, 10],
    "GroqModel": "whisper-large-v3-turbo",
    "GroqAccuracyModel": "whisper-large-v3",
    "GroqFallbackModel": "whisper-large-v3",
    "GroqApiKeyEnv": "GROQ_API_KEY",
    "GroqPrompt": "",
    "TextCleanupEnabled": False,
    "TextCleanupModel": "llama-3.1-8b-instant",
    "TextCleanupMaxTokens": 256,
    "TextCleanupPrompt": DEFAULT_TEXT_CLEANUP_PROMPT,
    "WindowsRecognitionCulture": "en-US",
    "WindowsRecognitionTimeoutSeconds": 30,
    "Language": "en",
    "RestoreClipboard": False,
    "PreloadModelOnEngineStart": True,
    "WarmupModelOnEngineStart": False,
    "StreamingTranscription": False,
    "StreamChunkSeconds": 6.0,
    "MinimumAudioSeconds": 0.6,
    "MinimumAudioRms": 0.00035,
    "PostProcessPunctuation": True,
    "SmartFormattingEnabled": False,
    "PhraseCorrections": DEFAULT_PHRASE_CORRECTIONS,
    "SubmitOnPressEnterCommand": True,
}

WINDOWS_SPEECH_RECOGNITION_SCRIPT = r"""
param(
    [Parameter(Mandatory=$true)][string]$WavPath,
    [string]$Culture = "en-US"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$recognizer = $null
try {
    if (-not [string]::IsNullOrWhiteSpace($Culture)) {
        $recognizer = [System.Speech.Recognition.SpeechRecognitionEngine]::new(
            [System.Globalization.CultureInfo]::new($Culture)
        )
    }
} catch {
    $recognizer = $null
}
if ($null -eq $recognizer) {
    $recognizer = [System.Speech.Recognition.SpeechRecognitionEngine]::new()
}

try {
    $recognizer.LoadGrammar([System.Speech.Recognition.DictationGrammar]::new())
    $recognizer.SetInputToWaveFile($WavPath)
    $parts = New-Object System.Collections.Generic.List[string]
    while ($true) {
        try {
            $result = $recognizer.Recognize()
        } catch [System.InvalidOperationException] {
            if ($parts.Count -gt 0) {
                break
            }
            throw
        }
        if ($null -eq $result) {
            break
        }
        if (-not [string]::IsNullOrWhiteSpace($result.Text)) {
            $parts.Add($result.Text.Trim())
        }
    }
    if ($parts.Count -gt 0) {
        [Console]::Out.Write([string]::Join(" ", $parts))
    }
} finally {
    if ($null -ne $recognizer) {
        $recognizer.Dispose()
    }
}
"""

OBSOLETE_CONFIG_KEYS = {
    "LlmCleanupEnabled",
    "LlmCleanupModel",
    "LlmCleanupMaxTokens",
    "LlmCleanupPrompt",
    "MuteMusicWhileDictating",
}


def log(message: str) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def acquire_engine_instance_lock() -> bool:
    global ENGINE_MUTEX_HANDLE
    if os.name != "nt":
        return True

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_bool

    handle = kernel32.CreateMutexW(None, False, "Local\\FDvoiceDictationEngine")
    if not handle:
        log(f"Engine single-instance mutex failed: {ctypes.WinError(ctypes.get_last_error())}")
        return True

    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        log("Another FDvoice dictation engine is already running; exiting duplicate engine")
        kernel32.CloseHandle(handle)
        return False

    ENGINE_MUTEX_HANDLE = handle
    return True


def configure_cuda_dll_paths() -> None:
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
    ]
    existing_dirs = [str(path) for path in dll_dirs if path.exists()]
    if not existing_dirs:
        return

    current_path = os.environ.get("PATH", "")
    for path in existing_dirs:
        if path not in current_path:
            current_path = path + os.pathsep + current_path
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError) as exc:
            log(f"CUDA DLL directory registration skipped for {path}: {type(exc).__name__}: {exc}")
    os.environ["PATH"] = current_path


def load_config() -> dict:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            log(f"Config parse failed, using defaults: {exc}")
            loaded = {}
    else:
        loaded = {}

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    if int(loaded.get("ConfigVersion", 0) or 0) < CONFIG_VERSION:
        config["ConfigVersion"] = CONFIG_VERSION
        for key in MIGRATED_DEFAULT_KEYS:
            config[key] = DEFAULT_CONFIG[key]
    for key in OBSOLETE_CONFIG_KEYS:
        config.pop(key, None)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def is_windows_backend(backend: str) -> bool:
    return str(backend).strip().lower() in ("windows", "system-speech", "system.speech", "sapi")


def should_post_process_transcript(config: dict, backend: str) -> bool:
    if is_windows_backend(backend):
        return False
    return bool(config.get("PostProcessPunctuation", True))


def should_run_text_cleanup(config: dict, backend: str) -> bool:
    return str(backend).strip().lower() == "groq" and bool(config.get("TextCleanupEnabled", False))


def resolve_groq_transcription_model(config: dict) -> str:
    mode = str(config.get("DictationMode", DEFAULT_CONFIG["DictationMode"])).strip().lower()
    if mode == "accuracy":
        return str(config.get("GroqAccuracyModel") or DEFAULT_CONFIG["GroqAccuracyModel"])
    return str(config.get("GroqModel") or DEFAULT_CONFIG["GroqModel"])


def normalize_cleanup_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def is_safe_cleanup_result(raw_text: str, cleaned_text: str) -> bool:
    cleaned = cleaned_text.strip()
    if not cleaned:
        return False

    filler_words = {"um", "uh", "ah", "hmm", "er", "eh"}
    raw_words = [word for word in normalize_cleanup_words(raw_text) if word not in filler_words]
    cleaned_words = [word for word in normalize_cleanup_words(cleaned_text) if word not in filler_words]
    if not cleaned_words:
        return False

    raw_counts = {}
    for word in raw_words:
        raw_counts[word] = raw_counts.get(word, 0) + 1

    cleaned_counts = {}
    for word in cleaned_words:
        cleaned_counts[word] = cleaned_counts.get(word, 0) + 1

    for word, count in cleaned_counts.items():
        if count > raw_counts.get(word, 0):
            return False
    return True


def send_virtual_key(vk_code: int) -> None:
    user32 = ctypes.windll.user32
    down = INPUT(type=INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(vk_code, 0, 0, 0, None)))
    up = INPUT(type=INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(vk_code, 0, KEYEVENTF_KEYUP, 0, None)))
    sent = user32.SendInput(2, (INPUT * 2)(down, up), ctypes.sizeof(INPUT))
    if sent != 2:
        raise ctypes.WinError(ctypes.get_last_error())


class MediaPauser:
    ACTIVE_SESSION_STATE = 1

    def __init__(self, process_names, sessions_provider=None, key_sender=None):
        self.process_names = {name.lower() for name in process_names}
        self.sessions_provider = sessions_provider or self._get_audio_sessions
        self.key_sender = key_sender or (lambda: send_virtual_key(VK_MEDIA_PLAY_PAUSE))
        self.paused_by_fdvoice = False

    def pause(self):
        self.paused_by_fdvoice = False
        if self._has_active_target_session():
            self.key_sender()
            self.paused_by_fdvoice = True
            log("Paused active media session")
        else:
            log("No active media session found to pause")

    def resume(self):
        if not self.paused_by_fdvoice:
            return
        self.key_sender()
        self.paused_by_fdvoice = False
        log("Resumed media session paused by FDvoice")

    def _get_audio_sessions(self):
        if AudioUtilities is None:
            log("pycaw unavailable; cannot detect active media sessions")
            return []
        return AudioUtilities.GetAllSessions()

    def _has_active_target_session(self) -> bool:
        try:
            if comtypes is not None:
                comtypes.CoInitialize()
            for session in self.sessions_provider():
                process = session.Process
                if process is None:
                    continue
                process_name = process.name().lower()
                if process_name.endswith(".exe"):
                    process_name = process_name[:-4]
                if self._is_target(process_name) and int(getattr(session, "State", 0)) == self.ACTIVE_SESSION_STATE:
                    log(f"Active media session detected: {process_name}")
                    return True
        except Exception as exc:
            log(f"Media session detection failed: {type(exc).__name__}: {exc}")
        return False

    def _is_target(self, process_name: str) -> bool:
        if process_name in self.process_names:
            return True
        return "youtube music" in process_name and (
            "youtube music" in self.process_names or "YouTube Music".lower() in self.process_names
        )


class TextInjector:
    @staticmethod
    def inject(target_hwnd: int, text: str, restore_clipboard: bool):
        user32 = ctypes.windll.user32
        if target_hwnd:
            user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.12)

        try:
            TextInjector._send_unicode_text(text)
            log("Text injected with SendInput unicode")
            return
        except Exception as exc:
            log(f"Unicode text injection failed; using clipboard fallback: {type(exc).__name__}: {exc}")

        TextInjector._paste_via_clipboard(text, restore_clipboard=True)

    @staticmethod
    def _paste_via_clipboard(text: str, restore_clipboard: bool):
        original = None
        if restore_clipboard:
            try:
                original = pyperclip.paste()
            except Exception:
                original = None

        pyperclip.copy(text)
        time.sleep(0.08)
        TextInjector._send_ctrl_v()
        time.sleep(0.5)

        if restore_clipboard and original is not None:
            try:
                pyperclip.copy(original)
            except Exception as exc:
                log(f"Clipboard restore failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _utf16_units(text: str) -> list[int]:
        encoded = text.encode("utf-16-le")
        return [int.from_bytes(encoded[index : index + 2], "little") for index in range(0, len(encoded), 2)]

    @staticmethod
    def _input(vk, flags):
        item = INPUT()
        item.type = INPUT_KEYBOARD
        item.union.ki = KEYBDINPUT(vk, 0, flags, 0, None)
        return item

    @staticmethod
    def _unicode_input(unit: int, flags: int):
        item = INPUT()
        item.type = INPUT_KEYBOARD
        item.union.ki = KEYBDINPUT(0, unit, flags | KEYEVENTF_UNICODE, 0, None)
        return item

    @staticmethod
    def _send_unicode_text(text: str):
        if not text:
            return
        user32 = ctypes.windll.user32
        user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
        user32.SendInput.restype = ctypes.c_uint
        units = TextInjector._utf16_units(text)
        events = []
        for unit in units:
            events.append(TextInjector._unicode_input(unit, 0))
            events.append(TextInjector._unicode_input(unit, KEYEVENTF_KEYUP))

        chunk_size = 256
        input_size = ctypes.sizeof(INPUT)
        for index in range(0, len(events), chunk_size):
            chunk = events[index : index + chunk_size]
            array_type = INPUT * len(chunk)
            sent = user32.SendInput(len(chunk), array_type(*chunk), input_size)
            if sent != len(chunk):
                raise ctypes.WinError(ctypes.get_last_error())
            time.sleep(0.01)

    @staticmethod
    def _send_ctrl_v():
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_V, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    @staticmethod
    def release_replay_modifiers():
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

    @staticmethod
    def send_enter():
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_RETURN, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)


def describe_window(hwnd: int) -> str:
    if not hwnd:
        return "hwnd=0"
    user32 = ctypes.windll.user32
    title_buffer = ctypes.create_unicode_buffer(256)
    try:
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        title = title_buffer.value
    except Exception:
        title = ""
    return f"hwnd={hwnd} title={title!r}"


def resolve_injection_target(start_hwnd: int, release_hwnd: int) -> int:
    return start_hwnd or release_hwnd


def audio_is_usable(audio: np.ndarray, min_duration: float, min_rms: float) -> bool:
    if len(audio) == 0:
        return False
    duration = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return duration >= min_duration and rms >= min_rms


def join_transcript_parts(parts: list[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned).replace("  ", " ").strip()


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['][A-Za-z0-9]+)?", text))


FILLER_WORDS = {"uh", "um", "er", "ah", "hmm"}
LEADING_DISCOURSE_WORDS = (
    "uh",
    "um",
    "er",
    "ah",
    "hmm",
    "oh",
    "okay",
    "ok",
    "so",
    "and",
    "well",
    "actually",
    "i mean",
)
TRAILING_DISCOURSE_PHRASES = (
    "uh",
    "um",
    "er",
    "ah",
    "hmm",
    "i mean",
    "you know",
    "you know what i mean",
    "and",
    "so",
    "okay",
    "ok",
    "well",
)
HESITATION_ONLY_WORDS = {
    "uh",
    "um",
    "er",
    "ah",
    "hmm",
    "oh",
    "and",
    "so",
    "well",
    "okay",
    "ok",
    "i",
    "mean",
    "ha",
    "haha",
    "huh",
}
REVISION_RESET_PATTERNS = (
    r"\b(?:scratch|forget)\s+that[\s,;:.-]*",
    r"\b(?:never\s+mind|nevermind)[\s,;:.-]*",
    r"\b(?:no|nah)[\s,;:.-]+(?:actually|wait|sorry)[\s,;:.-]*",
    r"\b(?:wait|sorry)[\s,;:.-]+(?:actually[\s,;:.-]+)?",
)
CHANGE_OBJECT_REVISION_PATTERN = re.compile(
    r"(?is)^(?P<prefix>.*?\b(?:i\s+want(?:\s+you)?\s+to\s+change|we\s+should\s+change|change)\s+)"
    r"(?P<old>[^.?!]+?)[.?!]\s*actually,\s*"
    r"(?P<reset>never\s+mind|nevermind|don['’]?t\s+mind|do\s+not\s+mind)[,.]?\s*"
    r"(?P<suggestion>how\s+about\s+we\s+change|let['’]?s\s+do|let\s+us\s+do|do)\s+"
    r"(?P<new>[^.?!]+)[.?!]?\s*$"
)


def build_history_entry(
    raw_text: str,
    final_text: str,
    audio_duration_seconds: float,
    transcription_seconds: float,
    cleanup_seconds: float,
    backend: str,
    submit_command: bool,
) -> dict:
    word_count = count_words(final_text)
    minutes = audio_duration_seconds / 60 if audio_duration_seconds > 0 else 0
    words_per_minute = round(word_count / minutes, 1) if minutes else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": backend,
        "raw_text": raw_text,
        "final_text": final_text,
        "word_count": word_count,
        "raw_word_count": count_words(raw_text),
        "audio_duration_ms": round(audio_duration_seconds * 1000),
        "transcription_ms": round(transcription_seconds * 1000),
        "cleanup_ms": round(cleanup_seconds * 1000),
        "words_per_minute": words_per_minute,
        "submit_command": submit_command,
    }


def append_history_entry(entry: dict):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_latest_history_final_text() -> str:
    if not HISTORY_PATH.exists():
        return ""
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        log(f"History read failed: {type(exc).__name__}: {exc}")
        return ""

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        text = str(entry.get("final_text") or "").strip()
        if text:
            return text
    return ""


def extract_press_enter_command(text: str) -> tuple[str, bool]:
    match = re.search(r"(?i)(?:^|\s)press\s+enter[\s.!,?]*$", text)
    if not match:
        return text, False
    return text[: match.start()].strip(" \t\r\n.,!?"), True


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_repeated_leading_discourse(text: str) -> str:
    choices = "|".join(re.escape(word) for word in sorted(LEADING_DISCOURSE_WORDS, key=len, reverse=True))
    pattern = re.compile(rf"(?i)^\s*(?:{choices})\b[\s,;:.!?-]*")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub("", text, count=1)
    return text


def _strip_trailing_discourse(text: str) -> str:
    choices = "|".join(re.escape(phrase) for phrase in sorted(TRAILING_DISCOURSE_PHRASES, key=len, reverse=True))
    pattern = re.compile(rf"(?i)(?:[\s,;:.!?-]+(?:{choices})\b[\s,;:.!?-]*)+$")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub("", text, count=1)
    return text.strip(" \t\r\n,;:-")


def _apply_revision_resets(text: str) -> str:
    matches = []
    for pattern in REVISION_RESET_PATTERNS:
        matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return text

    latest = max(matches, key=lambda match: match.end())
    candidate = text[latest.end() :].strip(" \t\r\n,;:.!?-")
    if count_words(candidate) == 0:
        return text
    return candidate


def _remove_filler_words(text: str) -> str:
    choices = "|".join(re.escape(word) for word in sorted(FILLER_WORDS, key=len, reverse=True))
    return re.sub(rf"(?i)(?:^|[\s,;:])\b(?:{choices})\b[,\s;:]*", " ", text)


def _strip_most_commas(text: str) -> str:
    return re.sub(r"\s*,+\s*", " ", text)


def _remove_trailing_rejected_alternative(text: str) -> str:
    return re.sub(
        r"(?i)\s*,?\s+instead\s+of\s+"
        r"(?:that|this|it|what\s+i\s+said|the\s+first\s+one|the\s+old\s+one|that\s+one|this\s+one)"
        r"[\s.!?]*$",
        "",
        text,
    )


def _normalize_leading_okay(text: str, keep_okay: bool) -> str:
    if keep_okay:
        return re.sub(r"(?i)^okay,\s*", "okay ", text)
    return re.sub(r"(?i)^okay,\s*", "", text)


def _apply_change_object_revision(text: str) -> tuple[str, bool]:
    match = CHANGE_OBJECT_REVISION_PATTERN.match(text)
    if not match:
        return text, False

    suggestion = match.group("suggestion").lower()
    keep_okay = not suggestion.startswith("how about")
    rewritten = match.group("prefix") + match.group("new")
    rewritten = _normalize_leading_okay(rewritten, keep_okay=keep_okay)
    rewritten = rewritten.strip(" \t\r\n,;:.!?-")
    return _normalize_spaces(rewritten), True


def _apply_phrase_corrections(text: str, phrase_corrections: dict | None = None) -> str:
    corrections = phrase_corrections if isinstance(phrase_corrections, dict) else DEFAULT_PHRASE_CORRECTIONS
    for source, replacement in sorted(corrections.items(), key=lambda item: len(str(item[0])), reverse=True):
        source_text = str(source).strip()
        if not source_text:
            continue
        text = re.sub(rf"(?i)\b{re.escape(source_text)}\b", str(replacement), text)
    return text


def _is_hesitation_only(text: str) -> bool:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z]+", text)]
    if not tokens:
        return True
    return all(token in HESITATION_ONLY_WORDS for token in tokens)


def _apply_explicit_quote_commands(text: str) -> str:
    return re.sub(
        r"(?i)\b(?:open\s+quote|quote)\s+([^.!?]+?)\s+(?:close\s+quote|end\s+quote)\b",
        lambda match: '"' + _normalize_spaces(match.group(1)) + '"',
        text,
    )


def _apply_explicit_parentheses_commands(text: str) -> str:
    return re.sub(
        r"(?i)\b(?:open|left)\s+parenthes(?:is|es)\s+([^.!?]+?)\s+(?:close|right)\s+parenthes(?:is|es)\b",
        lambda match: "(" + _normalize_spaces(match.group(1)) + ")",
        text,
    )


def _quote_named_phrase(match: re.Match) -> str:
    lead = match.group("lead")
    phrase = _normalize_spaces(match.group("phrase")).strip(' "\'')
    if not phrase or count_words(phrase) > 5:
        return match.group(0)
    return f'{lead} "{phrase}"'


def _apply_named_phrase_quotes(text: str) -> str:
    return re.sub(
        r"(?i)(?P<lead>\b(?:called|named|titled|labeled|labelled)\s+)"
        r"(?P<phrase>(?![\"']).{1,60}?)(?=$|[,.!?;:])",
        _quote_named_phrase,
        text,
    )


def apply_smart_formatting(text: str) -> str:
    text = _apply_explicit_quote_commands(text)
    text = _apply_explicit_parentheses_commands(text)
    text = _apply_named_phrase_quotes(text)
    text = re.sub(r"\s+([)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return _normalize_spaces(text)


def soften_punctuation(text: str) -> str:
    text = text.replace("\u2026", ".")
    text = text.replace("\u00e2\u20ac\u00a6", ".")
    text = re.sub(r"\s*[-\u2013\u2014]+\s*", " ", text)
    text = re.sub(r"(?<=[A-Za-z])\s*\.{2,}\s+(?=[a-z])", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])\1+", r"\1", text)
    return _normalize_spaces(text)


def minimize_punctuation(text: str) -> str:
    text = re.sub(r"(?<=\w)\.\s+(?=\w)", " ", text)
    text = re.sub(r"\.+$", "", text)
    text = re.sub(r"!+$", "!", text)
    text = re.sub(r"(?<=\w)!\s+(?=\w)", " ", text)
    return _normalize_spaces(text)


def deterministic_format_transcript(
    text: str,
    phrase_corrections: dict | None = None,
    smart_formatting: bool = True,
) -> str:
    text = _normalize_spaces(text)
    if not text:
        return ""

    text, was_object_revision = _apply_change_object_revision(text)
    if not was_object_revision:
        text = _apply_revision_resets(text)
        text = _strip_repeated_leading_discourse(text)
    text = _remove_filler_words(text)
    text = _strip_trailing_discourse(text)
    text = _remove_trailing_rejected_alternative(text)
    text = _strip_most_commas(text)
    text = soften_punctuation(text)
    text = _apply_phrase_corrections(text, phrase_corrections)
    if smart_formatting:
        text = apply_smart_formatting(text)
    text = _strip_trailing_discourse(text)
    text = soften_punctuation(text)
    text = minimize_punctuation(text)
    if _is_hesitation_only(text):
        return ""
    return text


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class WhisperDictationEngine:
    def __init__(self, config):
        self.config = config
        self.media_pauser = MediaPauser(config["MusicProcessNames"])
        self.model = None
        self.processor = None
        self.groq_client = None
        self.backend = str(config.get("ASRBackend", DEFAULT_CONFIG["ASRBackend"])).lower()
        self.recording = False
        self.target_hwnd = 0
        self.audio_chunks = []
        self.all_audio_chunks = []
        self.audio_buffer_samples = 0
        self.audio_lock = threading.Lock()
        self.stream_queue = None
        self.stream_results = []
        self.stream_seq = 0
        self.stream_worker = None
        self.stream = None
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.last_transcript_lock = threading.Lock()
        self.last_transcript = ""
        self.model_warmed = False
        self.streaming_enabled = bool(config.get("StreamingTranscription", True)) and not is_windows_backend(self.backend)
        self.stream_chunk_samples = int(float(config.get("StreamChunkSeconds", 3.0)) * SAMPLE_RATE)
        self.min_audio_seconds = float(config.get("MinimumAudioSeconds", DEFAULT_CONFIG["MinimumAudioSeconds"]))
        self.min_audio_rms = float(config.get("MinimumAudioRms", DEFAULT_CONFIG["MinimumAudioRms"]))

    def load_model(self):
        with self.model_lock:
            if self.model is not None:
                return

            if is_windows_backend(self.backend):
                self.backend = "windows"
                self.model = "windows-system-speech"
                log("Windows System.Speech backend ready")
                return

            if self.backend == "cohere":
                try:
                    self._load_cohere_model()
                    return
                except Exception as exc:
                    fallback_backend = str(
                        self.config.get(
                            "CohereFallbackBackend",
                            DEFAULT_CONFIG["CohereFallbackBackend"],
                        )
                        or "groq"
                    ).lower()
                    if fallback_backend == "cohere":
                        raise
                    log(
                        "Primary Cohere model load failed: "
                        f"{type(exc).__name__}: {exc}; falling back to {fallback_backend}"
                    )
                    self.backend = fallback_backend

            if is_windows_backend(self.backend):
                self.backend = "windows"
                self.model = "windows-system-speech"
                log("Windows System.Speech backend ready")
                return

            if self.backend == "groq":
                try:
                    self._load_groq_client()
                    return
                except Exception as exc:
                    log(f"Primary Groq client load failed: {type(exc).__name__}: {exc}")
                    self.backend = "faster-whisper"

            if self.backend == "parakeet":
                try:
                    self._load_parakeet_model()
                    return
                except Exception as exc:
                    log(f"Primary Parakeet model load failed: {type(exc).__name__}: {exc}")
                    self.backend = "faster-whisper"

            self._load_whisper_model()

    def _load_groq_client(self):
        from groq import Groq

        env_name = self.config.get("GroqApiKeyEnv", DEFAULT_CONFIG["GroqApiKeyEnv"])
        api_key = self._read_api_key(env_name)
        if not api_key:
            raise RuntimeError(f"{env_name} is not set")
        self.groq_client = Groq(api_key=api_key)
        self.model = self.groq_client
        self.backend = "groq"
        log(f"Groq client ready model={self.config.get('GroqModel', DEFAULT_CONFIG['GroqModel'])}")

    def _load_cohere_model(self):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_name = self.config.get("CohereModel", DEFAULT_CONFIG["CohereModel"])
        log(f"Loading Cohere ASR model={model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        self.backend = "cohere"
        log("Cohere ASR model loaded")

    def _read_api_key(self, env_name: str) -> str | None:
        value = os.environ.get(env_name)
        if value:
            return value
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                registry_value, _ = winreg.QueryValueEx(key, env_name)
            return registry_value
        except Exception:
            return None

    def _load_parakeet_model(self):
        import torch
        from nemoasr2pytorch.asr.api import load_parakeet_tdt_bf16, load_parakeet_tdt_fp16

        lang = self.config.get("ParakeetLang", DEFAULT_CONFIG["ParakeetLang"])
        device = self.config.get("ParakeetDevice", DEFAULT_CONFIG["ParakeetDevice"])
        dtype = str(self.config.get("ParakeetDType", DEFAULT_CONFIG["ParakeetDType"])).lower()
        if str(device).lower() == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for Parakeet")

        log(f"Loading Parakeet model lang={lang} device={device} dtype={dtype}")
        if dtype == "bfloat16":
            self.model = load_parakeet_tdt_bf16(device=device, lang=lang)
        else:
            self.model = load_parakeet_tdt_fp16(device=device, lang=lang)
        self.backend = "parakeet"
        log("Parakeet model loaded")

    def _load_whisper_model(self):
            if WhisperModel is None:
                raise RuntimeError("faster-whisper is not installed")
            model_name = self.config.get("WhisperModel", DEFAULT_CONFIG["WhisperModel"])
            device = self.config.get("WhisperDevice", DEFAULT_CONFIG["WhisperDevice"])
            compute_type = self.config.get("WhisperComputeType", DEFAULT_CONFIG["WhisperComputeType"])
            if str(device).lower() == "cuda":
                configure_cuda_dll_paths()
            log(f"Loading faster-whisper model={model_name} device={device} compute_type={compute_type}")
            try:
                self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                log("Whisper model loaded")
                return
            except Exception as exc:
                log(f"Primary Whisper model load failed: {type(exc).__name__}: {exc}")

            fallback_model = self.config.get("FallbackWhisperModel", DEFAULT_CONFIG["FallbackWhisperModel"])
            fallback_device = self.config.get("FallbackWhisperDevice", DEFAULT_CONFIG["FallbackWhisperDevice"])
            fallback_compute = self.config.get(
                "FallbackWhisperComputeType",
                DEFAULT_CONFIG["FallbackWhisperComputeType"],
            )
            log(
                "Loading fallback faster-whisper model="
                + str(fallback_model)
                + " device="
                + str(fallback_device)
                + " compute_type="
                + str(fallback_compute)
            )
            self.model = WhisperModel(fallback_model, device=fallback_device, compute_type=fallback_compute)
            self.backend = "faster-whisper"
            log("Fallback Whisper model loaded")

    def warmup_model(self):
        if self.model_warmed:
            return
        self.load_model()
        start_time = time.time()
        if is_windows_backend(self.backend):
            self.model_warmed = True
            log("Windows backend ready; skipping model warmup")
            return
        if self.backend == "groq":
            self.model_warmed = True
            log("Groq backend ready; skipping paid warmup request")
            return
        if self.backend == "parakeet":
            for seconds in self.config.get("ParakeetWarmupSeconds", DEFAULT_CONFIG["ParakeetWarmupSeconds"]):
                samples = max(SAMPLE_RATE, int(float(seconds) * SAMPLE_RATE))
                warmup_audio = np.zeros(samples, dtype=np.float32)
                self._transcribe_audio(warmup_audio)
        else:
            warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            segments, _ = self.model.transcribe(
                warmup_audio,
                language=self.config.get("Language", "en"),
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            for _ in segments:
                pass
        self.model_warmed = True
        log(f"{self.backend} model warmed in {time.time() - start_time:.2f}s")

    def preload_model(self):
        try:
            self.load_model()
            if self.config.get("WarmupModelOnEngineStart", True):
                self.warmup_model()
        except Exception as exc:
            log(f"Model preload failed: {type(exc).__name__}: {exc}")

    def begin_recording(self):
        with self.lock:
            if not self.recording:
                self.start()

    def end_recording(self):
        with self.lock:
            if self.recording:
                self.stop()

    def start(self):
        log("Start dictation requested")
        self.target_hwnd = ctypes.windll.user32.GetForegroundWindow()
        log(f"Dictation start focus target: {describe_window(self.target_hwnd)}")
        with self.audio_lock:
            self.audio_chunks = []
            self.all_audio_chunks = []
            self.audio_buffer_samples = 0
        self.stream_results = []
        self.stream_seq = 0
        if self.streaming_enabled:
            self._start_stream_worker()

        try:
            if self.config.get("PauseMediaWhileDictating", True):
                self.media_pauser.pause()

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.recording = True
            log("Recording started")
        except Exception as exc:
            self.recording = False
            self.stream = None
            self._stop_stream_worker()
            self.media_pauser.resume()
            log(f"Recording start failed: {type(exc).__name__}: {exc}")

    def _start_stream_worker(self):
        self.stream_queue = queue.Queue()
        self.stream_worker = threading.Thread(target=self._stream_worker_loop, daemon=True)
        self.stream_worker.start()
        log(f"Streaming worker started chunk_seconds={self.stream_chunk_samples / SAMPLE_RATE:.2f}")

    def _stream_worker_loop(self):
        while True:
            item = self.stream_queue.get()
            if item is None:
                return
            seq, audio = item
            if not audio_is_usable(audio, self.min_audio_seconds, self.min_audio_rms):
                log(f"Streaming chunk skipped seq={seq}")
                continue
            start_time = time.time()
            try:
                text = self._transcribe_audio(audio)
                with self.results_lock:
                    self.stream_results.append((seq, text, time.time() - start_time))
                log(
                    f"Streaming chunk transcribed seq={seq} duration={len(audio) / SAMPLE_RATE:.2f}s "
                    f"elapsed={time.time() - start_time:.2f}s text={text!r}"
                )
            except Exception as exc:
                log(f"Streaming chunk failed seq={seq}: {type(exc).__name__}: {exc}")

    def _enqueue_stream_chunk(self, audio: np.ndarray):
        if self.stream_queue is None or len(audio) == 0:
            return
        seq = self.stream_seq
        self.stream_seq += 1
        self.stream_queue.put((seq, audio))
        log(f"Streaming chunk queued seq={seq} duration={len(audio) / SAMPLE_RATE:.2f}s")

    def stop(self):
        log("Stop dictation requested")
        release_hwnd = ctypes.windll.user32.GetForegroundWindow()
        log(f"Dictation release focus target: {describe_window(release_hwnd)}")
        injection_hwnd = resolve_injection_target(self.target_hwnd, release_hwnd)
        log(f"Dictation injection target: {describe_window(injection_hwnd)}")
        self.recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.media_pauser.resume()

        with self.audio_lock:
            tail_chunks = self.audio_chunks
            self.audio_chunks = []
            self.audio_buffer_samples = 0
            full_chunks = self.all_audio_chunks
            self.all_audio_chunks = []

        if not full_chunks:
            log("No audio captured")
            self._stop_stream_worker()
            return

        audio = np.concatenate(full_chunks).astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        log(f"Captured audio duration={duration:.2f}s rms={rms:.5f}")
        if not audio_is_usable(audio, self.min_audio_seconds, self.min_audio_rms):
            log("Audio skipped: too short or too quiet")
            self._stop_stream_worker()
            return

        try:
            self.load_model()
            transcription_seconds = 0.0
            if self.streaming_enabled and self.stream_worker is not None:
                if tail_chunks:
                    tail = np.concatenate(tail_chunks).astype(np.float32)
                    self._enqueue_stream_chunk(tail)
                self._stop_stream_worker()
                with self.results_lock:
                    sorted_results = sorted(self.stream_results, key=lambda row: row[0])
                    text = join_transcript_parts(part for _, part, _ in sorted_results)
                    transcription_seconds = sum(elapsed for _, _, elapsed in sorted_results)
                if not text:
                    log("Streaming produced no text; falling back to full transcription")
                    transcription_start = time.time()
                    text = self._transcribe_audio(audio)
                    transcription_seconds = time.time() - transcription_start
            else:
                transcription_start = time.time()
                text = self._transcribe_audio(audio)
                transcription_seconds = time.time() - transcription_start
            raw_text = text
            cleanup_seconds = 0.0
            if should_run_text_cleanup(self.config, self.backend):
                cleanup_start = time.time()
                text = self._cleanup_text_with_groq(text)
                cleanup_seconds = time.time() - cleanup_start
            if should_post_process_transcript(self.config, self.backend):
                cleanup_start = time.time()
                text = deterministic_format_transcript(
                    text,
                    self.config.get("PhraseCorrections"),
                    bool(self.config.get("SmartFormattingEnabled", DEFAULT_CONFIG["SmartFormattingEnabled"])),
                )
                cleanup_seconds += time.time() - cleanup_start
            text, should_press_enter = extract_press_enter_command(text)
            if should_press_enter:
                text, _ = extract_press_enter_command(text)
            log(f"Transcription backend={self.backend} text={text!r}")
            if not text and not should_press_enter:
                log("No transcription text produced")
                return
            history_entry = build_history_entry(
                raw_text=raw_text,
                final_text=text,
                audio_duration_seconds=duration,
                transcription_seconds=transcription_seconds,
                cleanup_seconds=cleanup_seconds,
                backend=self.backend,
                submit_command=bool(should_press_enter and self.config.get("SubmitOnPressEnterCommand", True)),
            )
            append_history_entry(history_entry)
            log(
                "History entry written "
                f"words={history_entry['word_count']} wpm={history_entry['words_per_minute']} "
                f"transcription_ms={history_entry['transcription_ms']} cleanup_ms={history_entry['cleanup_ms']}"
            )
            if text:
                self._remember_transcript(text)
                TextInjector.inject(injection_hwnd, text, bool(self.config.get("RestoreClipboard", True)))
                log("Text injected")
            if should_press_enter and self.config.get("SubmitOnPressEnterCommand", True):
                time.sleep(0.08)
                TextInjector.send_enter()
                log("Press-enter command executed")
        except Exception as exc:
            log(f"Transcription/injection failed: {type(exc).__name__}: {exc}")

    def _remember_transcript(self, text: str):
        with self.last_transcript_lock:
            self.last_transcript = text
        log("Last transcript updated")

    def repeat_last_transcript(self):
        with self.last_transcript_lock:
            text = self.last_transcript
        history_text = read_latest_history_final_text()
        if history_text:
            text = history_text
        if not text:
            log("Repeat-last requested but no transcript is available")
            return
        target_hwnd = ctypes.windll.user32.GetForegroundWindow()
        TextInjector.release_replay_modifiers()
        time.sleep(0.05)
        TextInjector.inject(target_hwnd, text, bool(self.config.get("RestoreClipboard", True)))
        log("Last transcript repeated")

    def _stop_stream_worker(self):
        if self.stream_queue is not None:
            self.stream_queue.put(None)
        if self.stream_worker is not None:
            self.stream_worker.join(timeout=30)
            if self.stream_worker.is_alive():
                log("Streaming worker did not stop within timeout")
        self.stream_queue = None
        self.stream_worker = None

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        if is_windows_backend(self.backend):
            return self._transcribe_windows_audio(audio)

        if self.backend == "cohere":
            return self._transcribe_cohere_audio(audio)

        if self.backend == "groq":
            return self._transcribe_groq_audio(audio)

        if self.backend == "parakeet":
            import torch
            from nemoasr2pytorch.asr.api import transcribe_amp

            text = transcribe_amp(self.model, audio)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return str(text).strip()

        vad_parameters = {
            "min_silence_duration_ms": int(
                self.config.get("WhisperVadMinSilenceMs", DEFAULT_CONFIG["WhisperVadMinSilenceMs"])
            ),
            "speech_pad_ms": int(self.config.get("WhisperVadSpeechPadMs", DEFAULT_CONFIG["WhisperVadSpeechPadMs"])),
        }
        segments, info = self.model.transcribe(
            audio,
            language=self.config.get("Language", "en"),
            beam_size=3,
            vad_filter=True,
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,
            no_speech_threshold=float(
                self.config.get("WhisperNoSpeechThreshold", DEFAULT_CONFIG["WhisperNoSpeechThreshold"])
            ),
            log_prob_threshold=float(
                self.config.get("WhisperLogProbThreshold", DEFAULT_CONFIG["WhisperLogProbThreshold"])
            ),
            compression_ratio_threshold=float(
                self.config.get(
                    "WhisperCompressionRatioThreshold",
                    DEFAULT_CONFIG["WhisperCompressionRatioThreshold"],
                )
            ),
            hallucination_silence_threshold=float(
                self.config.get(
                    "WhisperHallucinationSilenceThreshold",
                    DEFAULT_CONFIG["WhisperHallucinationSilenceThreshold"],
                )
            ),
            hotwords=self.config.get("WhisperHotwords") or None,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        log(f"Whisper language={info.language} probability={info.language_probability:.3f}")
        return text

    def _transcribe_cohere_audio(self, audio: np.ndarray) -> str:
        self.load_model()
        punctuation = bool(self.config.get("CoherePunctuation", DEFAULT_CONFIG["CoherePunctuation"]))
        max_new_tokens = int(self.config.get("CohereMaxNewTokens", DEFAULT_CONFIG["CohereMaxNewTokens"]))
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=self.config.get("Language", "en"),
            punctuation=punctuation,
        )
        if hasattr(inputs, "to"):
            inputs.to(getattr(self.model, "device", None), dtype=getattr(self.model, "dtype", None))
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(outputs, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0] if text else ""
        log(f"Cohere transcription model={self.config.get('CohereModel', DEFAULT_CONFIG['CohereModel'])}")
        return str(text).strip()

    def _cleanup_text_with_groq(self, text: str) -> str:
        raw_text = text.strip()
        if not raw_text:
            return ""

        self.load_model()
        prompt_template = str(
            self.config.get("TextCleanupPrompt")
            or DEFAULT_CONFIG["TextCleanupPrompt"]
            or DEFAULT_TEXT_CLEANUP_PROMPT
        )
        prompt = prompt_template.replace("{{transcript}}", raw_text)
        if "{{transcript}}" not in prompt_template:
            prompt = prompt_template.rstrip() + "\n\nText:\n" + raw_text

        model = str(self.config.get("TextCleanupModel") or DEFAULT_CONFIG["TextCleanupModel"])
        max_tokens = int(self.config.get("TextCleanupMaxTokens", DEFAULT_CONFIG["TextCleanupMaxTokens"]))
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            cleaned = str(response.choices[0].message.content or "").strip()
        except Exception as exc:
            log(f"Groq text cleanup failed: {type(exc).__name__}: {exc}")
            return raw_text

        if not is_safe_cleanup_result(raw_text, cleaned):
            log(f"Groq text cleanup rejected unsafe output raw={raw_text!r} cleaned={cleaned!r}")
            return raw_text

        log(f"Groq text cleanup model={model}")
        return cleaned

    def _transcribe_windows_audio(self, audio: np.ndarray) -> str:
        if os.name != "nt":
            raise RuntimeError("Windows speech recognition backend is only available on Windows")

        wav_bytes = audio_to_wav_bytes(audio)
        timeout = float(
            self.config.get(
                "WindowsRecognitionTimeoutSeconds",
                DEFAULT_CONFIG["WindowsRecognitionTimeoutSeconds"],
            )
        )
        culture = str(
            self.config.get(
                "WindowsRecognitionCulture",
                DEFAULT_CONFIG["WindowsRecognitionCulture"],
            )
            or ""
        )
        with tempfile.NamedTemporaryFile(prefix="fdvoice-", suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)
            wav_file.write(wav_bytes)
        with tempfile.NamedTemporaryFile(
            prefix="fdvoice-windows-speech-",
            suffix=".ps1",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as script_file:
            script_path = Path(script_file.name)
            script_file.write(WINDOWS_SPEECH_RECOGNITION_SCRIPT)

        start_time = time.time()
        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-WavPath",
                    str(wav_path),
                    "-Culture",
                    culture,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                error_text = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(error_text or f"PowerShell exited with code {completed.returncode}")
            text = completed.stdout.strip()
            log(f"Windows System.Speech transcription elapsed={time.time() - start_time:.2f}s")
            return text
        finally:
            try:
                wav_path.unlink()
            except OSError as exc:
                log(f"Temporary Windows transcription WAV cleanup failed: {type(exc).__name__}: {exc}")
            try:
                script_path.unlink()
            except OSError as exc:
                log(f"Temporary Windows transcription script cleanup failed: {type(exc).__name__}: {exc}")

    def _transcribe_groq_audio(self, audio: np.ndarray) -> str:
        self.load_model()
        model = resolve_groq_transcription_model(self.config)
        wav_bytes = audio_to_wav_bytes(audio)
        start_time = time.time()
        try:
            response = self.groq_client.audio.transcriptions.create(
                file=("fdvoice.wav", wav_bytes),
                model=model,
                language=self.config.get("Language", "en"),
                prompt=self.config.get("GroqPrompt") or None,
                response_format="json",
                temperature=0,
            )
            text = response.text.strip()
            log(f"Groq transcription model={model} elapsed={time.time() - start_time:.2f}s")
            return text
        except Exception as exc:
            fallback_model = self.config.get("GroqFallbackModel", DEFAULT_CONFIG["GroqFallbackModel"])
            if fallback_model and fallback_model != model:
                log(f"Groq primary failed, trying fallback {fallback_model}: {type(exc).__name__}: {exc}")
                response = self.groq_client.audio.transcriptions.create(
                    file=("fdvoice.wav", wav_bytes),
                    model=fallback_model,
                    language=self.config.get("Language", "en"),
                    prompt=self.config.get("GroqPrompt") or None,
                    response_format="json",
                    temperature=0,
                )
                text = response.text.strip()
                log(f"Groq fallback transcription model={fallback_model} elapsed={time.time() - start_time:.2f}s")
                return text
            raise

    def _audio_callback(self, indata, frames, callback_time, status):
        if status:
            log(f"Audio callback status: {status}")
        if self.recording:
            chunk = indata[:, 0].copy()
            stream_chunk = None
            with self.audio_lock:
                self.audio_chunks.append(chunk)
                self.all_audio_chunks.append(chunk)
                self.audio_buffer_samples += len(chunk)
                if self.streaming_enabled and self.audio_buffer_samples >= self.stream_chunk_samples:
                    stream_chunk = np.concatenate(self.audio_chunks).astype(np.float32)
                    self.audio_chunks = []
                    self.audio_buffer_samples = 0
            if stream_chunk is not None:
                self._enqueue_stream_chunk(stream_chunk)


class HotkeyController:
    def __init__(self, engine):
        self.engine = engine
        self.ctrl_down = False
        self.win_down = False
        self.alt_down = False
        self.shift_down = False
        self.active = False
        self.replay_active = False

    def on_press(self, key):
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl_down = True
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.win_down = True
        if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.alt_down = True
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift_down = True
        if self.ctrl_down and self.win_down and not self.active:
            self.active = True
            log("Ctrl+Windows hold started")
            threading.Thread(target=self.engine.begin_recording, daemon=True).start()
        if self.alt_down and self.shift_down and self._is_replay_key(key) and not self.replay_active:
            self.replay_active = True
            log("Alt+Shift+Z repeat-last requested")
            threading.Thread(target=self.engine.repeat_last_transcript, daemon=True).start()

    def on_release(self, key):
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl_down = False
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            self.win_down = False
        if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.alt_down = False
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift_down = False
        if self.active and (not self.ctrl_down or not self.win_down):
            self.active = False
            log("Ctrl+Windows hold released")
            threading.Thread(target=self.engine.end_recording, daemon=True).start()
        if self.replay_active and (not self.alt_down or not self.shift_down or self._is_replay_key(key)):
            self.replay_active = False

    @staticmethod
    def _is_replay_key(key) -> bool:
        if isinstance(key, keyboard.KeyCode):
            return key.vk in (VK_V, VK_Z) or (key.char is not None and key.char.lower() in ("v", "z"))
        return False


def main():
    config = load_config()
    if not acquire_engine_instance_lock():
        return
    log("FDvoice dictation engine starting")
    log(
        "Config backend="
        + str(config.get("ASRBackend"))
        + " groq_model="
        + str(config.get("GroqModel"))
        + " whisper_model="
        + str(config.get("WhisperModel"))
    )
    engine = WhisperDictationEngine(config)
    controller = HotkeyController(engine)
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()
    log("Ctrl+Windows hold-to-dictate listener started; Alt+Shift+Z repeats last transcript")
    if config.get("PreloadModelOnEngineStart", True):
        threading.Thread(target=engine.preload_model, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Engine interrupted")
    finally:
        if engine.recording:
            engine.stop()


if __name__ == "__main__":
    main()
