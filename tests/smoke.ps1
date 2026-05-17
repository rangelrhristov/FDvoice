Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$sourceEngine = Join-Path $root "fdvoice-tauri\src-tauri\resources\fdvoice_whisper.py"
$unitTests = Join-Path $root "fdvoice-tauri\tests\test_fdvoice_cleanup.py"

$pythonCandidates = @(
    (Join-Path $root ".venv-parakeet\Scripts\python.exe"),
    (Join-Path $root ".venv\Scripts\python.exe"),
    "python.exe"
)

$python = $pythonCandidates | Where-Object { ($_ -eq "python.exe") -or (Test-Path $_) } | Select-Object -First 1
if (!$python) {
    throw "Python was not found for FDvoice smoke tests"
}

foreach ($path in @($sourceEngine, $unitTests)) {
    if (!(Test-Path $path)) {
        throw "Required FDvoice file is missing: $path"
    }
}

$content = Get-Content -Raw $sourceEngine
foreach ($required in @("WhisperDictationEngine", "HotkeyController", "ASRBackend", "CohereLabs/cohere-transcribe-03-2026", "CoherePunctuation", "CohereAsrForConditionalGeneration", "deterministic_format_transcript")) {
    if ($content -notmatch [regex]::Escape($required)) {
        throw "FDvoice engine does not contain required implementation marker: $required"
    }
}

& $python -m py_compile $sourceEngine
if ($LASTEXITCODE -ne 0) {
    throw "FDvoice engine py_compile failed with exit code $LASTEXITCODE"
}

& $python -m unittest $unitTests
if ($LASTEXITCODE -ne 0) {
    throw "FDvoice unit tests failed with exit code $LASTEXITCODE"
}

Write-Host "Smoke tests passed."
