// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//! Side-car backend lifecycle: locating, configuring and spawning the
//! PyInstaller-frozen `geti-backend` next to the Tauri executable.

use std::{
    env,
    path::PathBuf,
    process::{Child, Command},
};

/// "geti-backend.exe" on Windows, "geti-backend" elsewhere.
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
pub fn spawn_backend() -> std::io::Result<Child> {
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");
    let backend_path = exe_dir.join(backend_filename());

    log::info!("▶ Looking for backend side-car at {:?}", backend_path);
    let mut command = Command::new(&backend_path);
    apply_default_env(&mut command);

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let child = command.spawn()?;

    log::info!("▶ Spawned backend: {:?}", backend_path);
    Ok(child)
}

/// Apply the default environment for the side-car. Each var is only set if the
/// inherited environment doesn't already provide one, so callers can override.
fn apply_default_env(command: &mut Command) {
    // The Tauri 2 webview loads the UI from `tauri://localhost` on macOS/Linux
    // and `https://tauri.localhost` on Windows (Edge WebView2). Both must be
    // in the backend's CORS allowlist or every fetch from the UI is rejected.
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

    // Pin matplotlib's font/style cache to a stable per-user dir so it's built
    // once and reused across launches. Without this, matplotlib falls back to
    // a path inside the frozen `_internal/` (re-extracted on every launch),
    // forcing a full font-cache rebuild every start.
    if env::var_os("MPLCONFIGDIR").is_none() {
        if let Some(cache_dir) = mpl_cache_dir() {
            command.env("MPLCONFIGDIR", cache_dir);
        }
    }
}

/// Per-user matplotlib cache dir. Falls back to `None` if no suitable home
/// dir is available, in which case matplotlib uses its own default.
fn mpl_cache_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    let mut path = PathBuf::from(home);
    if cfg!(target_os = "macos") {
        path.push("Library/Caches/com.intel.geti/matplotlib");
    } else if cfg!(windows) {
        path.push("AppData/Local/com.intel.geti/matplotlib");
    } else {
        path.push(".cache/com.intel.geti/matplotlib");
    }
    let _ = std::fs::create_dir_all(&path);
    Some(path)
}
