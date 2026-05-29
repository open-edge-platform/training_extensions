// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use std::process::Child;
use std::sync::{Arc, Mutex};

use tauri::{Manager, RunEvent, WindowEvent};

use crate::backend::spawn_backend;

/// Kill a process and all its descendants.
///
/// - **Windows**: `taskkill /F /T /PID` terminates the entire process tree.
/// - **Unix**: sends `SIGKILL` to the process group (`kill -- -<pid>`).  The
///   backend inherits the Tauri-created process group so all its multiprocessing
///   workers are included.
fn kill_process_tree(child: &mut Child) {
    let pid = child.id();

    #[cfg(windows)]
    {
        use std::process::Command;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }

    #[cfg(unix)]
    {
        use std::process::Command;
        // kill -- -PID sends the signal to the whole process group.
        let _ = Command::new("kill")
            .args(["-9", "--", &format!("-{pid}")])
            .output();
    }

    // Reap the main child so we don't leave a zombie.
    let _ = child.wait();
}

fn shutdown_backend(child_handle: &Arc<Mutex<Option<Child>>>) {
    if let Some(mut child) = child_handle.lock().unwrap().take() {
        kill_process_tree(&mut child);
        log::info!("⛔ Backend terminated");
    }
}

fn main() {
    // Shared handle so we can kill the backend on exit.
    let child_handle = Arc::new(Mutex::new(None));

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_log::Builder::default().build())
        .setup({
            let child_handle = child_handle.clone();
            move |app| {
                let child = spawn_backend(app.handle()).expect("Failed to spawn python backend");
                *child_handle.lock().unwrap() = Some(child);
                Ok(())
            }
        })
        // Geti is a single-window utility app, so closing the main window
        // should quit the whole process (default macOS behaviour is to keep
        // the app alive in the dock, which leaks the backend side-car).
        .on_window_event({
            let child_handle = child_handle.clone();
            move |window, event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    // Prevent the default close so we can shut down gracefully.
                    // Destroying the window first lets the WebView2 / Chromium
                    // widget tear down cleanly before the process exits,
                    // avoiding the "Failed to unregister class
                    // Chrome_WidgetWin_0" error on Windows.
                    api.prevent_close();

                    // Kill the backend *before* exiting so worker processes
                    // cannot outlive the UI — even if RunEvent::Exit is
                    // short-circuited by exit(0).
                    shutdown_backend(&child_handle);

                    let handle = window.app_handle().clone();
                    if let Err(e) = window.destroy() {
                        log::warn!("Failed to destroy window during shutdown: {e}");
                    }
                    handle.exit(0);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![])
        .build(tauri::generate_context!())
        .expect("error building Tauri");

    // Belt-and-suspenders: also handle RunEvent::Exit for cases where the app
    // exits without going through the CloseRequested path (e.g. Cmd+Q on
    // macOS, or programmatic shutdown).
    let exit_handle = child_handle.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            shutdown_backend(&exit_handle);
        }
    });
}
