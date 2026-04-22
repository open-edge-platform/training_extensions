// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    env,
    process::{Child, Command},
    sync::{Arc, Mutex},
};
use tauri::{Manager, RunEvent, WindowEvent};

/// “geti-backend.exe” on Windows, “geti-backend” elsewhere.
fn backend_filename() -> &'static str {
    if cfg!(windows) {
        "geti-backend.exe"
    } else {
        "geti-backend"
    }
}

/// Spawns the side-car backend from the same folder as the Tauri executable.
///
/// Tauri stages `geti-backend` (from `externalBin`) and `_internal/` (from
/// `resources`) next to its own binary in both dev (`target/<profile>/`) and
/// release (the `.app` bundle), so PyInstaller's frozen layout works in both
/// modes without any extra path juggling.
fn spawn_backend() -> std::io::Result<Child> {
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");
    let backend_path = exe_dir.join(backend_filename());

    log::info!("▶ Looking for backend side-car at {:?}", backend_path);
    let mut command = Command::new(&backend_path);
    // The Tauri 2 webview loads the UI from `tauri://localhost` on macOS/Linux
    // and `https://tauri.localhost` on Windows (Edge WebView2). Both must be
    // in the backend's CORS allowlist or every fetch from the UI is rejected.
    // Only set a default — let the user override via the inherited environment.
    if env::var_os("CORS_ORIGINS").is_none() {
        // In `tauri dev` the webview loads the UI from the rsbuild dev server
        // at http://localhost:3000 (see tauri.conf.json `devUrl`), so that
        // origin must also be allowed or every fetch is blocked by CORS.
        #[cfg(debug_assertions)]
        let origins = "tauri://localhost,https://tauri.localhost,http://localhost:3000";
        #[cfg(not(debug_assertions))]
        let origins = "tauri://localhost,https://tauri.localhost";
        command.env("CORS_ORIGINS", origins);
    }
    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let child = command.spawn()?;

    log::info!("▶ Spawned backend: {:?}", backend_path);
    Ok(child)
}

fn main() {
    // Shared handle so we can kill it on exit
    let child_handle = Arc::new(Mutex::new(None));

    // Build the app
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup({
            let child_handle = child_handle.clone();
            move |_app_handle| {
                let child = spawn_backend().expect("Failed to spawn python backend");
                *child_handle.lock().unwrap() = Some(child);
                Ok(())
            }
        })
        // Geti is a single-window utility app, so closing the main window
        // should quit the whole process (default macOS behaviour is to keep
        // the app alive in the dock, which leaks the backend side-car).
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                window.app_handle().exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![])
        .build(tauri::generate_context!())
        .expect("error building Tauri");

    // Run and on Exit make sure to kill the backend
    let exit_handle = child_handle.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = exit_handle.lock().unwrap().take() {
                let _ = child.kill();
                log::info!("⛔ Backend terminated");
            }
        }
    });
}
