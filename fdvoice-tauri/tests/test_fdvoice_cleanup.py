import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src-tauri" / "resources" / "fdvoice_whisper.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fdvoice_whisper", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanupTests(unittest.TestCase):
    def test_default_config_uses_cohere_local_transcription_with_deterministic_cleanup(self):
        module = load_module()

        self.assertEqual(module.DEFAULT_CONFIG["ASRBackend"], "cohere")
        self.assertEqual(module.DEFAULT_CONFIG["CohereModel"], "CohereLabs/cohere-transcribe-03-2026")
        self.assertTrue(module.DEFAULT_CONFIG["CoherePunctuation"])
        self.assertEqual(module.DEFAULT_CONFIG["CohereFallbackBackend"], "groq")
        self.assertTrue(module.DEFAULT_CONFIG["PostProcessPunctuation"])
        self.assertFalse(module.DEFAULT_CONFIG["SmartFormattingEnabled"])
        self.assertFalse(module.DEFAULT_CONFIG["TextCleanupEnabled"])
        self.assertTrue(module.DEFAULT_CONFIG["PreloadModelOnEngineStart"])
        self.assertFalse(module.DEFAULT_CONFIG["WarmupModelOnEngineStart"])
        self.assertEqual(module.DEFAULT_CONFIG["GroqPrompt"], "")
        self.assertGreaterEqual(module.DEFAULT_CONFIG["MinimumAudioSeconds"], 0.6)

    def test_default_config_has_no_llm_cleanup_keys(self):
        module = load_module()

        self.assertNotIn("LlmCleanupEnabled", module.DEFAULT_CONFIG)
        self.assertNotIn("LlmCleanupModel", module.DEFAULT_CONFIG)
        self.assertNotIn("LlmCleanupMaxTokens", module.DEFAULT_CONFIG)
        self.assertNotIn("LlmCleanupPrompt", module.DEFAULT_CONFIG)

    def test_default_config_uses_full_context_transcription(self):
        module = load_module()

        self.assertFalse(module.DEFAULT_CONFIG["StreamingTranscription"])

    def test_load_config_removes_obsolete_llm_cleanup_keys_and_migrates_cohere_default(self):
        module = load_module()
        original_config_path = module.CONFIG_PATH
        temp_path = Path(__file__).with_name("tmp_config.json")
        try:
            temp_path.write_text(
                json.dumps(
                    {
                        "ConfigVersion": 1,
                        "LlmCleanupEnabled": True,
                        "LlmCleanupModel": "llama-3.1-8b-instant",
                        "LlmCleanupMaxTokens": 256,
                        "LlmCleanupPrompt": "rewrite this",
                    }
                ),
                encoding="utf-8",
            )
            module.CONFIG_PATH = temp_path

            config = module.load_config()

            self.assertNotIn("LlmCleanupEnabled", config)
            self.assertNotIn("LlmCleanupModel", config)
            self.assertNotIn("LlmCleanupMaxTokens", config)
            self.assertNotIn("LlmCleanupPrompt", config)
            self.assertEqual(config["ASRBackend"], "cohere")
            self.assertEqual(config["CohereModel"], "CohereLabs/cohere-transcribe-03-2026")
            self.assertTrue(config["CoherePunctuation"])
            self.assertEqual(config["CohereFallbackBackend"], "groq")
            self.assertFalse(config["TextCleanupEnabled"])
            self.assertTrue(config["PostProcessPunctuation"])
            self.assertFalse(config["SmartFormattingEnabled"])
            self.assertTrue(config["PreloadModelOnEngineStart"])
            self.assertFalse(config["WarmupModelOnEngineStart"])
            self.assertFalse(config["StreamingTranscription"])
        finally:
            module.CONFIG_PATH = original_config_path
            if temp_path.exists():
                temp_path.unlink()

    def test_windows_backend_bypasses_deterministic_formatter(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update(
            {
                "ASRBackend": "windows",
                "PostProcessPunctuation": True,
                "SmartFormattingEnabled": True,
            }
        )

        self.assertFalse(module.should_post_process_transcript(config, "windows"))

    def test_non_windows_backend_can_use_deterministic_formatter(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update(
            {
                "ASRBackend": "faster-whisper",
                "PostProcessPunctuation": True,
            }
        )

        self.assertTrue(module.should_post_process_transcript(config, "faster-whisper"))

    def test_cohere_transcription_uses_transformers_processor_and_generation(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update({"CoherePunctuation": False, "CohereMaxNewTokens": 77})

        class FakeInputs(dict):
            def __init__(self):
                super().__init__({"input_features": "features"})
                self.to_args = None

            def to(self, device, dtype=None):
                self.to_args = (device, dtype)
                return self

        class FakeProcessor:
            def __init__(self):
                self.inputs = FakeInputs()
                self.call_kwargs = None
                self.decode_kwargs = None

            def __call__(self, audio, **kwargs):
                self.call_kwargs = kwargs
                return self.inputs

            def decode(self, outputs, **kwargs):
                self.decode_kwargs = kwargs
                return " Um hello, hello, CPU, GPU, foo_bar: test?"

        class FakeModel:
            device = "cuda:0"
            dtype = "float16"

            def __init__(self):
                self.generate_kwargs = None

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return ["tokens"]

        processor = FakeProcessor()
        fake_model = FakeModel()
        engine = module.WhisperDictationEngine(config)
        engine.backend = "cohere"
        engine.processor = processor
        engine.model = fake_model

        result = engine._transcribe_audio(module.np.zeros(module.SAMPLE_RATE, dtype=module.np.float32))

        self.assertEqual(result, "Um hello, hello, CPU, GPU, foo_bar: test?")
        self.assertEqual(processor.call_kwargs["sampling_rate"], 16000)
        self.assertTrue(processor.call_kwargs["return_tensors"], "pt")
        self.assertEqual(processor.call_kwargs["language"], "en")
        self.assertFalse(processor.call_kwargs["punctuation"])
        self.assertEqual(processor.inputs.to_args, ("cuda:0", "float16"))
        self.assertEqual(fake_model.generate_kwargs["input_features"], "features")
        self.assertEqual(fake_model.generate_kwargs["max_new_tokens"], 77)
        self.assertTrue(processor.decode_kwargs["skip_special_tokens"])

    def test_cohere_load_failure_falls_back_to_configured_backend(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update({"ASRBackend": "cohere", "CohereFallbackBackend": "groq"})

        engine = module.WhisperDictationEngine(config)
        calls = []

        def fail_cohere():
            calls.append("cohere")
            raise OSError("gated repo")

        def load_groq():
            calls.append("groq")
            engine.model = "groq-client"
            engine.backend = "groq"

        engine._load_cohere_model = fail_cohere
        engine._load_groq_client = load_groq

        engine.load_model()

        self.assertEqual(calls, ["cohere", "groq"])
        self.assertEqual(engine.backend, "groq")
        self.assertEqual(engine.model, "groq-client")

    def test_cohere_cleanup_removes_fillers_and_most_commas_without_collapsing_repeats(self):
        module = load_module()

        result = module.deterministic_format_transcript(
            "um hello, hello, CPU, GPU, foo_bar: test?",
            smart_formatting=False,
        )

        self.assertEqual(result, "hello hello CPU GPU foo_bar: test?")

    def test_groq_default_mode_uses_turbo_transcription_model(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update({"DictationMode": "default"})

        self.assertEqual(module.resolve_groq_transcription_model(config), "whisper-large-v3-turbo")

    def test_groq_accuracy_mode_uses_large_v3_transcription_model(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update({"DictationMode": "accuracy"})

        self.assertEqual(module.resolve_groq_transcription_model(config), "whisper-large-v3")

    def test_groq_transcription_uses_resolved_mode_model(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update({"ASRBackend": "groq", "DictationMode": "accuracy"})

        class FakeTranscriptions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(text="hello")

        transcriptions = FakeTranscriptions()
        engine = module.WhisperDictationEngine(config)
        engine.backend = "groq"
        engine.groq_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
        engine.model = engine.groq_client

        result = engine._transcribe_groq_audio(module.np.zeros(module.SAMPLE_RATE, dtype=module.np.float32))

        self.assertEqual(result, "hello")
        self.assertEqual(transcriptions.kwargs["model"], "whisper-large-v3")

    def test_text_cleanup_prompt_matches_direct_insertion_rules(self):
        module = load_module()

        prompt = module.DEFAULT_TEXT_CLEANUP_PROMPT

        self.assertIn("Clean this dictated text for direct insertion.", prompt)
        self.assertIn("Remove filler words like um, uh, ah, hmm.", prompt)
        self.assertIn("Keep the wording casual.", prompt)
        self.assertIn("Return only the cleaned text.", prompt)

    def test_cleanup_validation_accepts_filler_removal_and_punctuation(self):
        module = load_module()

        self.assertTrue(
            module.is_safe_cleanup_result(
                "um can you fix this thing",
                "can you fix this thing.",
            )
        )

    def test_cleanup_validation_rejects_added_content(self):
        module = load_module()

        self.assertFalse(
            module.is_safe_cleanup_result(
                "can you fix this thing",
                "can you fix the database migration thing",
            )
        )

    def test_groq_text_cleanup_uses_prompt_and_model(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update(
            {
                "TextCleanupModel": "llama-3.1-8b-instant",
                "TextCleanupMaxTokens": 99,
            }
        )

        class FakeChoice:
            message = SimpleNamespace(content="can you fix this")

        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(choices=[FakeChoice()])

        completions = FakeCompletions()
        engine = module.WhisperDictationEngine(config)
        engine.backend = "groq"
        engine.groq_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        engine.model = engine.groq_client

        result = engine._cleanup_text_with_groq("um can you fix this")

        self.assertEqual(result, "can you fix this")
        self.assertEqual(completions.kwargs["model"], "llama-3.1-8b-instant")
        self.assertEqual(completions.kwargs["max_tokens"], 99)
        self.assertEqual(completions.kwargs["temperature"], 0)
        self.assertIn("Clean this dictated text for direct insertion.", completions.kwargs["messages"][0]["content"])
        self.assertIn("um can you fix this", completions.kwargs["messages"][0]["content"])

    def test_groq_text_cleanup_rejects_unsafe_response(self):
        module = load_module()

        class FakeChoice:
            message = SimpleNamespace(content="the database migration is broken")

        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[FakeChoice()])

        engine = module.WhisperDictationEngine(dict(module.DEFAULT_CONFIG))
        engine.backend = "groq"
        engine.groq_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        engine.model = engine.groq_client

        result = engine._cleanup_text_with_groq("this is broken")

        self.assertEqual(result, "this is broken")

    def test_media_pauser_pauses_and_resumes_active_target_session(self):
        module = load_module()
        sent_keys = []
        sessions = [
            SimpleNamespace(Process=SimpleNamespace(name=lambda: "chrome.exe"), State=1),
            SimpleNamespace(Process=SimpleNamespace(name=lambda: "discord.exe"), State=0),
        ]
        controller = module.MediaPauser(
            ["chrome"],
            sessions_provider=lambda: sessions,
            key_sender=lambda: sent_keys.append("play_pause"),
        )

        controller.pause()
        controller.resume()

        self.assertEqual(sent_keys, ["play_pause", "play_pause"])

    def test_media_pauser_does_not_resume_when_nothing_was_paused(self):
        module = load_module()
        sent_keys = []
        sessions = [SimpleNamespace(Process=SimpleNamespace(name=lambda: "chrome.exe"), State=0)]
        controller = module.MediaPauser(
            ["chrome"],
            sessions_provider=lambda: sessions,
            key_sender=lambda: sent_keys.append("play_pause"),
        )

        controller.pause()
        controller.resume()

        self.assertEqual(sent_keys, [])

    def test_deterministic_format_removes_fillers_and_softens_punctuation(self):
        module = load_module()

        result = module.deterministic_format_transcript(
            "uh can you fix this... um I think it broke!!!"
        )

        self.assertEqual(result, "can you fix this I think it broke!")

    def test_deterministic_format_removes_unneeded_sentence_periods(self):
        module = load_module()

        result = module.deterministic_format_transcript("This is a test. It should stay casual.")

        self.assertEqual(result, "This is a test It should stay casual")

    def test_deterministic_format_keeps_question_marks_and_colons(self):
        module = load_module()

        result = module.deterministic_format_transcript("Can you check this? Use this format: name value.")

        self.assertEqual(result, "Can you check this? Use this format: name value")

    def test_deterministic_format_preserves_intentional_repeated_adjacent_words(self):
        module = load_module()

        result = module.deterministic_format_transcript("testing testing testing I need this")

        self.assertEqual(result, "testing testing testing I need this")

    def test_deterministic_format_removes_repeated_fillers(self):
        module = load_module()

        result = module.deterministic_format_transcript("um um can you fix this")

        self.assertEqual(result, "can you fix this")

    def test_deterministic_format_handles_front_loaded_self_correction(self):
        module = load_module()

        result = module.deterministic_format_transcript("oh actually add this instead of that")

        self.assertEqual(result, "add this")

    def test_deterministic_format_handles_midstream_restart(self):
        module = load_module()

        result = module.deterministic_format_transcript("I want the blue one, no actually make it red")

        self.assertEqual(result, "make it red")

    def test_deterministic_format_rewrites_change_object_from_never_mind_suggestion(self):
        module = load_module()

        result = module.deterministic_format_transcript(
            "Okay, so basically I want you to change the UI. Actually, never mind. "
            "How about we change the settings?"
        )

        self.assertEqual(result, "so basically I want you to change the settings")

    def test_deterministic_format_rewrites_change_object_from_dont_mind_revision(self):
        module = load_module()

        result = module.deterministic_format_transcript(
            "Okay, so basically I want to change the UI. Actually, don't mind, let's do the settings."
        )

        self.assertEqual(result, "okay so basically I want to change the settings")

    def test_deterministic_format_preserves_meaningful_instead_of_phrase(self):
        module = load_module()

        result = module.deterministic_format_transcript("Use apples instead of oranges")

        self.assertEqual(result, "Use apples instead of oranges")

    def test_deterministic_format_discards_hesitation_only_output(self):
        module = load_module()

        result = module.deterministic_format_transcript("And... I mean...")

        self.assertEqual(result, "")

    def test_deterministic_format_discards_vocalization_only_output(self):
        module = load_module()

        result = module.deterministic_format_transcript("Ha ha ha ha.")

        self.assertEqual(result, "")

    def test_deterministic_format_applies_phrase_corrections(self):
        module = load_module()

        result = module.deterministic_format_transcript("This is FD voice, not EFT voice.")

        self.assertEqual(result, "This is FDvoice not FDvoice")

    def test_smart_formatting_handles_explicit_quote_commands(self):
        module = load_module()

        result = module.deterministic_format_transcript("Set the title to quote advanced settings end quote")

        self.assertEqual(result, 'Set the title to "advanced settings"')

    def test_smart_formatting_handles_explicit_parentheses_commands(self):
        module = load_module()

        result = module.deterministic_format_transcript("Use the fast model open parenthesis local close parenthesis")

        self.assertEqual(result, "Use the fast model (local)")

    def test_smart_formatting_quotes_named_short_phrases(self):
        module = load_module()

        result = module.deterministic_format_transcript("Create a section called advanced settings")

        self.assertEqual(result, 'Create a section called "advanced settings"')

    def test_faster_whisper_transcription_uses_noise_and_hotword_settings(self):
        module = load_module()
        config = dict(module.DEFAULT_CONFIG)
        config.update(
            {
                "WhisperHotwords": "FDvoice Codex",
                "WhisperNoSpeechThreshold": 0.7,
                "WhisperLogProbThreshold": -0.8,
                "WhisperCompressionRatioThreshold": 2.1,
                "WhisperHallucinationSilenceThreshold": 1.2,
                "WhisperVadMinSilenceMs": 450,
                "WhisperVadSpeechPadMs": 180,
            }
        )

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                return [SimpleNamespace(text=" hello ")], SimpleNamespace(
                    language="en",
                    language_probability=0.99,
                )

        fake_model = FakeModel()
        engine = module.WhisperDictationEngine(config)
        engine.backend = "faster-whisper"
        engine.model = fake_model

        result = engine._transcribe_audio(module.np.zeros(module.SAMPLE_RATE, dtype=module.np.float32))

        self.assertEqual(result, "hello")
        self.assertEqual(fake_model.kwargs["hotwords"], "FDvoice Codex")
        self.assertEqual(fake_model.kwargs["no_speech_threshold"], 0.7)
        self.assertEqual(fake_model.kwargs["log_prob_threshold"], -0.8)
        self.assertEqual(fake_model.kwargs["compression_ratio_threshold"], 2.1)
        self.assertEqual(fake_model.kwargs["hallucination_silence_threshold"], 1.2)
        self.assertEqual(fake_model.kwargs["vad_parameters"]["min_silence_duration_ms"], 450)
        self.assertEqual(fake_model.kwargs["vad_parameters"]["speech_pad_ms"], 180)

    def test_trailing_press_enter_command_is_removed_and_detected(self):
        module = load_module()

        text, should_press_enter = module.extract_press_enter_command(
            "Can you check this for me press enter."
        )

        self.assertEqual(text, "Can you check this for me")
        self.assertTrue(should_press_enter)

    def test_press_enter_command_only_matches_at_end(self):
        module = load_module()

        text, should_press_enter = module.extract_press_enter_command(
            "Can you press enter in this form later?"
        )

        self.assertEqual(text, "Can you press enter in this form later?")
        self.assertFalse(should_press_enter)

    def test_press_enter_command_does_not_match_send_it(self):
        module = load_module()

        text, should_press_enter = module.extract_press_enter_command("Send this send it")

        self.assertEqual(text, "Send this send it")
        self.assertFalse(should_press_enter)

    def test_count_words_handles_contractions_and_numbers(self):
        module = load_module()

        self.assertEqual(module.count_words("I don't have 2 minutes."), 5)

    def test_audio_is_usable_accepts_quiet_real_speech_level(self):
        module = load_module()
        audio = module.np.full(module.SAMPLE_RATE, 0.0008, dtype=module.np.float32)

        self.assertTrue(
            module.audio_is_usable(
                audio,
                module.DEFAULT_CONFIG["MinimumAudioSeconds"],
                module.DEFAULT_CONFIG["MinimumAudioRms"],
            )
        )

    def test_build_history_entry_records_metrics(self):
        module = load_module()

        entry = module.build_history_entry(
            raw_text="uh hello there",
            final_text="Hello there",
            audio_duration_seconds=2.5,
            transcription_seconds=0.42,
            cleanup_seconds=0.18,
            backend="groq",
            submit_command=False,
        )

        self.assertEqual(entry["raw_text"], "uh hello there")
        self.assertEqual(entry["final_text"], "Hello there")
        self.assertEqual(entry["word_count"], 2)
        self.assertEqual(entry["audio_duration_ms"], 2500)
        self.assertEqual(entry["transcription_ms"], 420)
        self.assertEqual(entry["cleanup_ms"], 180)
        self.assertEqual(entry["words_per_minute"], 48.0)
        self.assertFalse(entry["submit_command"])

    def test_injection_target_prefers_dictation_start_focus(self):
        module = load_module()

        self.assertEqual(module.resolve_injection_target(111, 222), 111)

    def test_injection_target_falls_back_to_release_focus(self):
        module = load_module()

        self.assertEqual(module.resolve_injection_target(0, 222), 222)

    def test_latest_history_final_text_reads_last_valid_entry(self):
        module = load_module()
        original_history_path = module.HISTORY_PATH
        temp_path = Path(__file__).with_name("tmp_history.jsonl")
        try:
            entries = [
                {"final_text": "First prompt"},
                {"final_text": "Most recent prompt"},
            ]
            temp_path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
            module.HISTORY_PATH = temp_path

            self.assertEqual(module.read_latest_history_final_text(), "Most recent prompt")
        finally:
            module.HISTORY_PATH = original_history_path
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
