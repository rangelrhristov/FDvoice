# FDvoice Tauri

This is the Tauri shell for FDvoice, a Windows dictation app with a local speech-to-text engine.

The Tauri app manages a bundled Windows dictation engine:

- Default hotkey: `Ctrl+Windows`
- Dictation engine: local Cohere Transcribe `CohereLabs/cohere-transcribe-03-2026` with deterministic cleanup by default
- Media pause: pauses active browser/music playback while dictating, then resumes it on release
- Injection: Unicode `SendInput` into the focused app, with clipboard fallback available
- Repeat last dictation: `Alt+Shift+Z`

## Run

```powershell
npm install
python -m pip install -r requirements.txt
npm run tauri:dev
```

The app starts hidden in the tray and launches the dictation engine automatically. Hold `Ctrl+Windows` anywhere in Windows while speaking, then release to insert the transcript.

For CUDA acceleration, install the PyTorch wheel that matches your GPU before installing the remaining Python dependencies.

## Build

Run from a Visual Studio Build Tools environment:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && npm run tauri:build"
```

## Verify

```powershell
npm run build
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && cd src-tauri && cargo test"
python -m unittest .\tests\test_fdvoice_cleanup.py
```
