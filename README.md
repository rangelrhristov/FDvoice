# FDvoice

FDvoice is a Windows-first dictation prototype with a Tauri desktop shell and a local speech-to-text engine.

- Hold `Ctrl+Windows` to dictate, then release to transcribe and insert.
- Press `Alt+Shift+Z` to repeat the last inserted dictation.
- While dictating, FDvoice can pause active browser/music playback, then resume it on release.
- When dictation stops, recognized text is pasted into the app that was focused when dictation started.
- Dictation defaults to local Cohere Transcribe `CohereLabs/cohere-transcribe-03-2026` through Hugging Face Transformers, then runs deterministic cleanup for direct insertion.

## Run

From the source app:

```powershell
cd fdvoice-tauri
npm install
python -m pip install -r requirements.txt
npm run tauri:dev
```

The app lives in the tray. Use the tray menu for config and exit.

Release binaries and installer outputs are build artifacts and are intentionally not committed.

## Configuration

On first run, FDvoice creates:

```text
%APPDATA%\FDvoice\config.json
```

Defaults:

- `HotkeyMode`: `Ctrl+Win`
- `ASRBackend`: `cohere`
- `CohereModel`: `CohereLabs/cohere-transcribe-03-2026`
- `CoherePunctuation`: `true`; set to `false` to ask the model for unpunctuated output
- `PostProcessPunctuation`: `true`
- `SmartFormattingEnabled`: `false`
- `PauseMediaWhileDictating`: `true`
- `MusicProcessNames`: includes `chrome`, `msedge`, `firefox`, `brave`, `spotify`, and `YouTube Music`

## Notes

The bundled runtime engine source is `fdvoice-tauri\src-tauri\resources\fdvoice_whisper.py`. Windows speech recognition, Faster-Whisper, Groq, and Parakeet backends remain configurable, but the default is local Cohere Transcribe.

Deterministic cleanup removes filler words, false starts, and most commas while preserving intentional repeated words, code-ish tokens, acronyms, periods, question marks, and colons where useful.

## Verify

```powershell
cd fdvoice-tauri
npm run build
python -m unittest .\tests\test_fdvoice_cleanup.py
```
