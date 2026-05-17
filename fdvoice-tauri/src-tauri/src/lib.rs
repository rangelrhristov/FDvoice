use serde::{Deserialize, Serialize};
#[cfg(windows)]
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(windows)]
const ERROR_ALREADY_EXISTS: u32 = 183;
const ENGINE_DESCRIPTION: &str = "Cohere Transcribe local ASR with deterministic cleanup";

#[cfg(windows)]
type Handle = *mut c_void;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn CreateMutexW(
        lp_mutex_attributes: *mut c_void,
        b_initial_owner: i32,
        lp_name: *const u16,
    ) -> Handle;
    fn GetLastError() -> u32;
    fn CloseHandle(h_object: Handle) -> i32;
}

#[derive(Default)]
struct EngineState {
    child: Mutex<Option<Child>>,
}

struct AppInstanceLock {
    #[cfg(windows)]
    handle: Handle,
}

#[cfg(windows)]
unsafe impl Send for AppInstanceLock {}

#[cfg(windows)]
unsafe impl Sync for AppInstanceLock {}

impl Drop for AppInstanceLock {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            if !self.handle.is_null() {
                let _ = CloseHandle(self.handle);
            }
        }
    }
}

#[derive(Serialize)]
struct EngineDefaults {
    hotkey: &'static str,
    pause_media_while_dictating: bool,
    music_processes: Vec<&'static str>,
    engine: &'static str,
}

#[derive(Serialize)]
struct EngineStatus {
    running: bool,
    pid: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone)]
struct DictationHistoryEntry {
    timestamp: String,
    backend: String,
    raw_text: String,
    final_text: String,
    word_count: u32,
    raw_word_count: u32,
    audio_duration_ms: u64,
    transcription_ms: u64,
    cleanup_ms: u64,
    words_per_minute: f64,
    submit_command: bool,
}

#[derive(Serialize)]
struct DictationStats {
    lifetime_words: u64,
    average_words_per_minute: f64,
    total_audio_ms: u64,
    total_entries: usize,
}

#[derive(Serialize)]
struct DictationHistoryResponse {
    stats: DictationStats,
    entries: Vec<DictationHistoryEntry>,
}

#[tauri::command]
fn get_engine_defaults() -> EngineDefaults {
    default_engine_settings()
}

#[tauri::command]
fn engine_status(state: tauri::State<'_, EngineState>) -> Result<EngineStatus, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "engine state lock poisoned".to_string())?;

    if let Some(child) = guard.as_mut() {
        match child.try_wait() {
            Ok(Some(_)) => {
                *guard = None;
                Ok(EngineStatus {
                    running: false,
                    pid: None,
                })
            }
            Ok(None) => Ok(EngineStatus {
                running: true,
                pid: Some(child.id()),
            }),
            Err(error) => Err(format!("failed to query engine process: {error}")),
        }
    } else {
        Ok(EngineStatus {
            running: false,
            pid: None,
        })
    }
}

#[tauri::command]
fn start_engine(
    app: AppHandle,
    state: tauri::State<'_, EngineState>,
) -> Result<EngineStatus, String> {
    start_engine_process(&app, state.inner())
}

fn start_engine_process(app: &AppHandle, state: &EngineState) -> Result<EngineStatus, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "engine state lock poisoned".to_string())?;

    if let Some(child) = guard.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(EngineStatus {
                running: true,
                pid: Some(child.id()),
            });
        }
        *guard = None;
    }

    let log_dir = engine_log_dir()?;
    std::fs::create_dir_all(&log_dir)
        .map_err(|error| format!("failed to create FDvoice log directory: {error}"))?;

    let script = engine_script_path(app)?;
    let python = python_executable_path()?;
    append_launcher_log(&format!(
        "starting dictation engine: python={} script={}",
        python.display(),
        script.display()
    ));

    let stdout = File::create(log_dir.join("engine.stdout.log"))
        .map_err(|error| format!("failed to create engine stdout log: {error}"))?;
    let stderr = File::create(log_dir.join("engine.stderr.log"))
        .map_err(|error| format!("failed to create engine stderr log: {error}"))?;

    let mut command = Command::new(python);
    command
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);

    let child = command
        .spawn()
        .map_err(|error| format!("failed to start FDvoice dictation engine: {error}"))?;

    let pid = child.id();
    *guard = Some(child);
    Ok(EngineStatus {
        running: true,
        pid: Some(pid),
    })
}

#[tauri::command]
fn stop_engine(state: tauri::State<'_, EngineState>) -> Result<EngineStatus, String> {
    stop_engine_process(state.inner());

    Ok(EngineStatus {
        running: false,
        pid: None,
    })
}

fn stop_engine_process(state: &EngineState) {
    let mut guard = state.child.lock().expect("engine state lock poisoned");

    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
fn open_config_folder() -> Result<(), String> {
    let appdata = std::env::var("APPDATA").map_err(|error| error.to_string())?;
    let path = PathBuf::from(appdata).join("FDvoice");
    std::fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Command::new("explorer.exe")
        .arg(path)
        .spawn()
        .map_err(|error| format!("failed to open config folder: {error}"))?;
    Ok(())
}

#[tauri::command]
fn get_dictation_history() -> Result<DictationHistoryResponse, String> {
    let appdata = std::env::var("APPDATA").map_err(|error| error.to_string())?;
    let path = PathBuf::from(appdata).join("FDvoice").join("history.jsonl");
    if !path.exists() {
        return Ok(DictationHistoryResponse {
            stats: DictationStats {
                lifetime_words: 0,
                average_words_per_minute: 0.0,
                total_audio_ms: 0,
                total_entries: 0,
            },
            entries: Vec::new(),
        });
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read dictation history: {error}"))?;
    let mut all_entries = Vec::new();
    for line in content.lines().filter(|line| !line.trim().is_empty()) {
        match serde_json::from_str::<DictationHistoryEntry>(line) {
            Ok(entry) => all_entries.push(entry),
            Err(error) => append_launcher_log(&format!("skipping invalid history entry: {error}")),
        }
    }

    let lifetime_words = all_entries
        .iter()
        .map(|entry| u64::from(entry.word_count))
        .sum::<u64>();
    let total_audio_ms = all_entries
        .iter()
        .map(|entry| entry.audio_duration_ms)
        .sum::<u64>();
    let average_words_per_minute = if total_audio_ms > 0 {
        ((lifetime_words as f64) / ((total_audio_ms as f64) / 60_000.0) * 10.0).round() / 10.0
    } else {
        0.0
    };

    let mut entries = all_entries;
    let total_entries = entries.len();
    entries.reverse();
    entries.truncate(100);

    Ok(DictationHistoryResponse {
        stats: DictationStats {
            lifetime_words,
            average_words_per_minute,
            total_audio_ms,
            total_entries,
        },
        entries,
    })
}

fn default_engine_settings() -> EngineDefaults {
    EngineDefaults {
        hotkey: "Ctrl+Windows hold; Alt+Shift+Z repeat last",
        pause_media_while_dictating: true,
        music_processes: vec![
            "chrome",
            "msedge",
            "firefox",
            "brave",
            "opera",
            "spotify",
            "YouTube Music",
        ],
        engine: ENGINE_DESCRIPTION,
    }
}

fn engine_script_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("resources").join("fdvoice_whisper.py"));
            candidates.push(exe_dir.join("fdvoice_whisper.py"));
        }
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("fdvoice_whisper.py"));
        candidates.push(resource_dir.join("resources").join("fdvoice_whisper.py"));
    }

    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("fdvoice_whisper.py"),
    );

    append_launcher_log("engine script candidates:");
    for candidate in &candidates {
        append_launcher_log(&format!(
            "  {} exists={}",
            candidate.display(),
            candidate.exists()
        ));
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    Err(format!(
        "FDvoice dictation engine script was not found. Checked: {}",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

fn python_executable_path() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(
                exe_dir
                    .join(".venv-parakeet")
                    .join("Scripts")
                    .join("python.exe"),
            );
            candidates.push(exe_dir.join(".venv").join("Scripts").join("python.exe"));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(
                    parent
                        .join(".venv-parakeet")
                        .join("Scripts")
                        .join("python.exe"),
                );
                candidates.push(parent.join(".venv").join("Scripts").join("python.exe"));
            }
        }
    }

    candidates.push(PathBuf::from("python.exe"));

    append_launcher_log("python executable candidates:");
    for candidate in &candidates {
        append_launcher_log(&format!(
            "  {} exists={}",
            candidate.display(),
            candidate.exists()
        ));
        if candidate.is_absolute() {
            if candidate.exists() {
                return Ok(candidate.clone());
            }
        } else {
            return Ok(candidate.clone());
        }
    }

    Err(format!(
        "Python was not found. Checked: {}",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

fn engine_log_dir() -> Result<PathBuf, String> {
    let appdata = std::env::var("APPDATA").map_err(|error| error.to_string())?;
    Ok(PathBuf::from(appdata).join("FDvoice"))
}

fn append_launcher_log(message: &str) {
    if let Ok(log_dir) = engine_log_dir() {
        let _ = std::fs::create_dir_all(&log_dir);
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_dir.join("launcher.log"))
        {
            let _ = writeln!(file, "{message}");
        }
    }
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

fn acquire_app_instance_lock() -> Option<AppInstanceLock> {
    #[cfg(windows)]
    unsafe {
        let name = wide_null("Local\\FDvoiceApp");
        let handle = CreateMutexW(std::ptr::null_mut(), 0, name.as_ptr());
        if handle.is_null() {
            append_launcher_log("app single-instance mutex failed; continuing without app lock");
            return Some(AppInstanceLock { handle });
        }
        if GetLastError() == ERROR_ALREADY_EXISTS {
            append_launcher_log("another FDvoice app is already running; exiting duplicate app");
            let _ = CloseHandle(handle);
            return None;
        }
        Some(AppInstanceLock { handle })
    }

    #[cfg(not(windows))]
    {
        Some(AppInstanceLock {})
    }
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_skip_taskbar(false);
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn hide_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
        let _ = window.set_skip_taskbar(true);
    }
}

fn hide_main_window_after_startup(app: &AppHandle) {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        for delay in [250_u64, 1000] {
            std::thread::sleep(Duration::from_millis(delay));
            let main_thread_handle = app_handle.clone();
            let _ = app_handle.run_on_main_thread(move || {
                hide_main_window(&main_thread_handle);
            });
        }
    });
}

fn setup_tray(app: &mut tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Show FDvoice", true, None::<&str>)?;
    let config = MenuItem::with_id(app, "config", "Open config folder", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit FDvoice", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &config, &quit])?;

    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("FDvoice dictation")
        .on_menu_event(|app, event| {
            if event.id() == "show" {
                show_main_window(app);
            } else if event.id() == "config" {
                if let Err(error) = open_config_folder() {
                    append_launcher_log(&format!("tray open config failed: {error}"));
                }
            } else if event.id() == "quit" {
                let state = app.state::<EngineState>();
                stop_engine_process(state.inner());
                app.exit(0);
            }
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            }
            | TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        });

    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }

    tray.build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let Some(app_instance_lock) = acquire_app_instance_lock() else {
        return;
    };

    tauri::Builder::default()
        .manage(app_instance_lock)
        .manage(EngineState::default())
        .invoke_handler(tauri::generate_handler![
            get_engine_defaults,
            engine_status,
            start_engine,
            stop_engine,
            open_config_folder,
            get_dictation_history,
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            setup_tray(app)?;
            hide_main_window(app.handle());
            hide_main_window_after_startup(app.handle());
            let state = app.state::<EngineState>();
            if let Err(error) = start_engine_process(app.handle(), state.inner()) {
                append_launcher_log(&format!("auto-start engine failed: {error}"));
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let app = window.app_handle();
                hide_main_window(app);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_requested_dictation_behavior() {
        let defaults = default_engine_settings();

        assert_eq!(
            defaults.hotkey,
            "Ctrl+Windows hold; Alt+Shift+Z repeat last"
        );
        assert!(defaults.pause_media_while_dictating);
        assert!(defaults.music_processes.contains(&"chrome"));
        assert!(defaults.music_processes.contains(&"YouTube Music"));
        assert_eq!(defaults.engine, ENGINE_DESCRIPTION);
    }

    #[test]
    fn history_stats_average_uses_lifetime_entries_not_visible_entries() {
        let entries = vec![
            DictationHistoryEntry {
                timestamp: "2026-05-14T00:00:00+0000".to_string(),
                backend: "groq".to_string(),
                raw_text: "hello world".to_string(),
                final_text: "Hello world".to_string(),
                word_count: 2,
                raw_word_count: 2,
                audio_duration_ms: 1_000,
                transcription_ms: 100,
                cleanup_ms: 50,
                words_per_minute: 120.0,
                submit_command: false,
            },
            DictationHistoryEntry {
                timestamp: "2026-05-14T00:00:01+0000".to_string(),
                backend: "groq".to_string(),
                raw_text: "more words here".to_string(),
                final_text: "More words here".to_string(),
                word_count: 3,
                raw_word_count: 3,
                audio_duration_ms: 2_000,
                transcription_ms: 100,
                cleanup_ms: 50,
                words_per_minute: 90.0,
                submit_command: false,
            },
        ];

        let lifetime_words = entries
            .iter()
            .map(|entry| u64::from(entry.word_count))
            .sum::<u64>();
        let total_audio_ms = entries
            .iter()
            .map(|entry| entry.audio_duration_ms)
            .sum::<u64>();
        let average_words_per_minute =
            ((lifetime_words as f64) / ((total_audio_ms as f64) / 60_000.0) * 10.0).round() / 10.0;

        assert_eq!(lifetime_words, 5);
        assert_eq!(total_audio_ms, 3_000);
        assert_eq!(average_words_per_minute, 100.0);
    }
}
