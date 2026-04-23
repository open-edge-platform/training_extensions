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

/// Spawns the side-car backend.
///
/// Layout depends on packaging:
///
/// - **`tauri dev`** stages `geti-backend` + `_internal/` flat next to the
///   Tauri executable in `target/<profile>/`. PyInstaller's bootloader sees
///   it's not inside a `.app` and uses the sibling `_internal/` directly.
/// - **`tauri build` (.app on macOS)** moves them into a `backend/`
///   subdirectory under `Contents/MacOS/`. This is required because if the
///   PyInstaller-frozen executable lives directly at
///   `<Bundle>.app/Contents/MacOS/<exe>` the bootloader switches to "macOS
///   bundle" mode and looks for shared libraries under `Contents/Frameworks/`
///   — which Tauri doesn't populate, so launch fails with
///   `Failed to load Python shared library 'libpython*.dylib'`. Putting the
///   sidecar one directory deeper avoids the bundle detection and keeps the
///   standalone `_internal/` layout working. The post-build move is done by
///   the packaging recipe in [`application/Justfile`](../../Justfile).
/// - **Other platforms** keep the flat layout in both dev and release because
///   no equivalent bundle-detection exists on Windows / Linux.
pub fn spawn_backend() -> std::io::Result<Child> {
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");
    let backend_path = locate_backend(exe_dir);

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

/// Resolve the side-car path. Prefers `<exe_dir>/backend/<name>` (release
/// `.app` layout) and falls back to `<exe_dir>/<name>` (dev / non-macOS).
fn locate_backend(exe_dir: &std::path::Path) -> PathBuf {
    let nested = exe_dir.join("backend").join(backend_filename());
    if nested.exists() {
        return nested;
    }
    exe_dir.join(backend_filename())
}

/// Apply the default environment for the side-car. Each var is only set if the
/// inherited environment doesn't already provide one, so callers can override.
fn apply_default_env(command: &mut Command) {
    // The Tauri 2 webview loads the UI from `tauri://localhost` on macOS/Linux
    // and `https://tauri.localhost` on Windows (Edge WebView2). Both must be
    // in the backend's CORS allowlist or every fetch from the UI is rejected.
    // Set this unconditionally — `CORS_ORIGINS` is a Tauri-context concern,
    // and a stale value in the user's shell or a `.env` file shadowing the
    // default would silently break the app with cryptic CORS errors.
    // In `tauri dev` the webview loads the UI from the rsbuild dev server
    // at http://localhost:3000 (see tauri.conf.json `devUrl`), so that
    // origin must also be allowed or every fetch is blocked by CORS.
    #[cfg(debug_assertions)]
    let cors_origins = "tauri://localhost,https://tauri.localhost,http://localhost:3000";
    #[cfg(not(debug_assertions))]
    let cors_origins = "tauri://localhost,https://tauri.localhost";
    command.env("CORS_ORIGINS", cors_origins);

    // Pin matplotlib's font/style cache to a stable per-user dir so it's built
    // once and reused across launches. Without this, matplotlib falls back to
    // a path inside the frozen `_internal/` (re-extracted on every launch),
    // forcing a full font-cache rebuild every start.
    if env::var_os("MPLCONFIGDIR").is_none() {
        if let Some(cache_dir) = per_user_subdir(CacheRoot::Cache, "matplotlib") {
            command.env("MPLCONFIGDIR", cache_dir);
        }
    }

    // The backend defaults `data_dir=Path("data")` and `log_dir=Path("logs")`
    // (relative to cwd). In a packaged `.app`/`Program Files` install that
    // directory is read-only, and in dev it pollutes `src-tauri/`. Pin both
    // to a per-user writable location so the same layout works in dev and
    // release without `.taurignore` masking the file-watcher loop.
    if env::var_os("DATA_DIR").is_none() {
        if let Some(dir) = per_user_subdir(CacheRoot::Data, "data") {
            command.env("DATA_DIR", dir);
        }
    }
    if env::var_os("LOG_DIR").is_none() {
        if let Some(dir) = per_user_subdir(CacheRoot::Data, "logs") {
            command.env("LOG_DIR", dir);
        }
    }
}

/// Distinguishes ephemeral cache from durable application data so we land in
/// the OS-conventional folder (matters most on macOS, where `~/Library/Caches`
/// is purgeable while `~/Library/Application Support` is not).
enum CacheRoot {
    Cache,
    Data,
}

/// Per-user `<root>/com.intel.geti/<sub>` directory, created if missing.
/// Returns `None` if no suitable home dir is available.
fn per_user_subdir(root: CacheRoot, sub: &str) -> Option<PathBuf> {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    let mut path = PathBuf::from(home);
    let base = match (cfg!(target_os = "macos"), cfg!(windows), &root) {
        (true, _, CacheRoot::Cache) => "Library/Caches",
        (true, _, CacheRoot::Data) => "Library/Application Support",
        (_, true, _) => "AppData/Local",
        (_, _, CacheRoot::Cache) => ".cache",
        (_, _, CacheRoot::Data) => ".local/share",
    };
    path.push(base);
    path.push("com.intel.geti");
    path.push(sub);
    let _ = std::fs::create_dir_all(&path);
    Some(path)
}
