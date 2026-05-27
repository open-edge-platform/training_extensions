# Geti Desktop (Tauri shell)

This folder contains the Tauri 2 wrapper that ships the Geti UI as a native
desktop application on macOS, Linux, and Windows. The web UI source lives one
level up in [`../src`](../src); the Rust shell lives in [`./src`](./src).

## Module resolution architecture

The same TypeScript source tree powers both the browser SPA and the Tauri
desktop build. Per-platform behaviour (downloading files, native menus,
auto-update, OS-specific styles, …) is selected **at build time by the
bundler**, not at runtime.

### How it works

`rsbuild.config.ts` reads `process.env.BUILD_TARGET`. When it equals `"tauri"`,
the Rspack `resolve.extensions` list is prepended with `.tauri.tsx`,
`.tauri.ts`, `.tauri.jsx`, `.tauri.js`, `.tauri.scss`:

```ts
// application/ui/rsbuild.config.ts
const isTauriBuild = process.env.BUILD_TARGET === 'tauri';
const platformExtensions = isTauriBuild
    ? ['.tauri.tsx', '.tauri.ts', '.tauri.jsx', '.tauri.js', '.tauri.scss']
    : [];

resolve: {
    extensions: [...platformExtensions, '.tsx', '.ts', '.jsx', '.js', '.json'],
}
```

A consumer always imports a plain name:

```ts
import { downloadFile } from './download-file';
```

The bundler resolves that import in extension order:

| Build         | Resolution order                 | File picked                     |
| ------------- | -------------------------------- | ------------------------------- |
| Web (default) | `.tsx`, `.ts`, …                 | `download-file.ts`              |
| Tauri         | `.tauri.tsx`, `.tauri.ts`, `.ts` | `download-file.tauri.ts` (wins) |

The unselected file is **not parsed and never enters the module graph**, so
`@tauri-apps/*` imports cannot leak into the web bundle and the web fallbacks
cannot bloat the desktop bundle.

### Conventions

```
src/
  platform/
    download-file.ts            ← web (default)
    download-file.tauri.ts      ← tauri override
```

Twins can live anywhere under `src/`; `src/platform/` is just where existing
capability modules are grouped.

Rules of thumb when adding a platform-specific behaviour:

1. **Plain file is the default.** It runs in both web and Tauri _unless_
   shadowed by a `.tauri.*` twin sitting next to it.
2. **`.tauri.*` files may import `@tauri-apps/*`.** Other source files may
   not — this is enforced by the `no-restricted-imports` rule in
   [`../eslint.config.js`](../eslint.config.js).
3. **Tauri-only features:** ship a no-op/null-returning module as the default
   and the real implementation in `.tauri.{ts,tsx}`. Consumers render/call
   unconditionally; the web build tree-shakes the no-op away.
4. **Tauri-only styles:** same trick with `.scss` / `.tauri.scss`. The
   `.tauri.scss` and `.scss` extensions are already in `resolve.extensions`,
   so an extensionless import (`import './foo'`) resolves to `foo.tauri.scss`
   on the desktop build and `foo.scss` on the web build. Drop the extension
   on the import site to opt in to the override.
5. **No `isTauri()` runtime checks anywhere.** Enforced by the
   `no-restricted-syntax` rule in [`../eslint.config.js`](../eslint.config.js).
   If you find yourself reaching for one, add (or split) a capability module
   instead.

The Tauri shell wires `BUILD_TARGET=tauri` for both the dev server and the
production build via `beforeDevCommand` / `beforeBuildCommand` in
[`tauri.conf.json`](./tauri.conf.json), which invoke the `start:tauri` and
`build:tauri` scripts in [`../package.json`](../package.json).

## Prerequisites (all platforms)

- **Node.js** ≥ 24.2 and **npm** ≥ 11.3 (see `engines` in `../package.json`).
- **Rust** stable toolchain ≥ 1.77.2 (install via [rustup](https://rustup.rs)).
- **`just`** task runner (used to build the Python backend; install via
  `brew install just`, `cargo install just`, `winget install Casey.Just`, or
  your package manager).
- **`bash`** on `PATH`. The backend `Justfile` uses bash heredocs in several
  recipes. macOS and Linux already provide it. On **Windows you must install
  [Git for Windows](https://git-scm.com/download/win)** and make sure its
  `cmd/` and `usr/bin/` folders are on `PATH` so `just` can resolve
  `bash.exe` and `cygpath.exe`. Without these, recipes such as
  `just pyinstaller` fail with `cygpath.exe: command not found` or
  `program not found` errors.
- **`uv`** Python package manager — installed automatically by the backend
  `Justfile` if missing, or via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Tauri 2 system dependencies** — see the platform-specific sections below.

Bootstrap the JS dependencies once from `application/ui`:

```sh
npm install
```

## Backend sidecar (required before any Tauri build)

The Tauri shell ships the FastAPI backend as a PyInstaller-frozen executable
side-car (`externalBin` in [`tauri.conf.json`](./tauri.conf.json)). It must
be built **once** before `tauri dev` / `tauri build` will succeed, and
rebuilt whenever the backend Python source changes.

### 1. Build the backend

From `application/backend`:

```sh
just pyinstaller            # CPU build
# or, for accelerated builds:
just pyinstaller -a xpu     # Intel GPU
just pyinstaller -a cuda    # NVIDIA GPU
```

On **macOS** the recipe transparently runs `just fix-macho-signatures`
first, which repairs malformed Mach-O dylibs shipped by some upstream wheels
(notably `openvino`'s `libhwloc` / `libtbb*`). Without this, ad-hoc
codesigning during PyInstaller's `COLLECT` phase fails with
_"internal error in Code Signing subsystem"_. The repair script lives at
[`../../backend/pyinstaller/fix_macho_signatures.py`](../../backend/pyinstaller/fix_macho_signatures.py)
and is a no-op on non-macOS platforms. If you ever need to run it on its
own (e.g. after a manual `uv sync`):

```sh
just fix-macho-signatures
```

On **Windows** you might need to install [Git for Windows](https://git-scm.com/download/win)
first — the backend `Justfile` uses bash heredocs, so `just` shells out to
`bash.exe` and `cygpath.exe`. Without them you'll see
`cygpath.exe: command not found` or `program not found`. During install,
accept the _"Git from the command line and also from 3rd-party software"_
option (or add `C:\Program Files\Git\cmd` and `C:\Program Files\Git\usr\bin`
to `PATH` manually), open a fresh PowerShell and verify:

```powershell
where.exe bash
where.exe cygpath
```

This produces `application/backend/dist/geti-backend/`, which contains:

- `geti-backend` (or `geti-backend.exe` on Windows) — the entry executable.
- `_internal/` — PyInstaller's bundled libraries, datas and Python runtime.

### 2. Wire the binary into `src-tauri/`

Tauri's `externalBin` mechanism requires the executable next to
`tauri.conf.json` and **named with the Rust target triple suffix** so
`tauri build` can locate it per platform. Both the binary and the
`_internal/` resources directory must be reachable from `src-tauri/`.

We use **symlinks** so the side-car always reflects the latest PyInstaller
run without copying gigabytes around.

cd into `application/ui/src-tauri`, then:

```sh
# macOS — Apple silicon (M1/M2/M3/M4)
ln -sf ../../backend/dist/geti-backend/geti-backend geti-backend-aarch64-apple-darwin
ln -sf ../../backend/dist/geti-backend/_internal _internal

# macOS — Intel
ln -sf ../../backend/dist/geti-backend/geti-backend geti-backend-x86_64-apple-darwin
ln -sf ../../backend/dist/geti-backend/_internal _internal

# Linux
ln -sf ../../backend/dist/geti-backend/geti-backend geti-backend-x86_64-unknown-linux-gnu
ln -sf ../../backend/dist/geti-backend/_internal _internal

# Windows (PowerShell, run as admin or with Developer Mode enabled)
New-Item -ItemType SymbolicLink -Path .\geti-backend-x86_64-pc-windows-msvc.exe `
    -Target ..\..\backend\dist\geti-backend\geti-backend.exe
New-Item -ItemType SymbolicLink -Path .\_internal `
    -Target ..\..\backend\dist\geti-backend\_internal
```

Find your host triple with `rustc -vV | grep host` if unsure.

The symlinks are gitignored. Tauri stages both `geti-backend` (from
`externalBin`) and `_internal/` (from `resources`) next to its own
executable in `target/<profile>/` during `tauri dev`, so PyInstaller's
frozen layout works in dev with no extra steps. The release `.app` on
macOS needs the sidecar nested one level deeper — that's done by the
`just tauri-build` recipe in [`../../Justfile`](../../Justfile); see
`spawn_backend()` and `locate_backend()` in [`src/backend.rs`](./src/backend.rs).

### macOS

1. Install Xcode Command Line Tools:
    ```sh
    xcode-select --install
    ```
2. (Recommended) Install [Homebrew](https://brew.sh) and Rust:
    ```sh
    brew install rustup-init && rustup-init
    ```
3. From `application/ui`, run the desktop dev shell:

    ```sh
    npm run start:desktop
    ```

    This invokes `tauri dev`, which in turn runs `npm run start:tauri` (sets
    `BUILD_TARGET=tauri` and starts the Rspack dev server) and launches the
    native window once the dev server is ready.

4. Build a distributable `.app` / `.dmg`:
    ```sh
    just tauri-build   # from application/
    ```
    Artifacts land in `application/ui/src-tauri/target/release/bundle/`.
    Note: the recipe runs `npx tauri build` and then patches the bundle
    layout (moves the sidecar + `_internal/` into `Contents/MacOS/backend/`)
    to work around PyInstaller's `.app` bundle detection. Running
    `npx tauri build` directly produces a `.app` whose backend dies on
    launch with `Failed to load Python shared library 'libpython*.dylib'`.

### Linux

1. Install the Tauri 2 system libraries (Debian/Ubuntu names shown; see the
   [Tauri prerequisites page](https://v2.tauri.app/start/prerequisites/) for
   Fedora/Arch/openSUSE equivalents):
    ```sh
    sudo apt update
    sudo apt install \
        libwebkit2gtk-4.1-dev \
        build-essential \
        curl \
        wget \
        file \
        libxdo-dev \
        libssl-dev \
        libayatana-appindicator3-dev \
        librsvg2-dev
    ```
2. Install Rust via [rustup](https://rustup.rs).
3. From `application/ui`, run the desktop dev shell:

    ```sh
    npm run start:desktop
    ```

4. Build a distributable AppImage / `.deb`:
    ```sh
    just tauri-build   # from application/
    ```
    Artifacts land in `application/ui/src-tauri/target/release/bundle/`.

### Windows

1. Install [Microsoft Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the **Desktop development with C++** workload (provides the MSVC
   linker required by Rust).
2. Install [WebView2 runtime](https://developer.microsoft.com/microsoft-edge/webview2/)
   (already present on Windows 11 and most up-to-date Windows 10 installs).
3. Install Rust via [rustup](https://rustup.rs).
4. From `application/ui` (PowerShell or `cmd`), run the desktop dev shell:

    ```powershell
    npm run start:desktop
    ```

5. Build a distributable `.msi` / `.exe`:
    ```powershell
    just tauri-build   # from application\
    ```
    Artifacts land in `application\ui\src-tauri\target\release\bundle\`.

> The `start:tauri` and `build:tauri` npm scripts set `BUILD_TARGET=tauri`
> using POSIX shell syntax (`BUILD_TARGET=tauri rsbuild …`). On native
> Windows shells this works because npm runs scripts through a POSIX-ish
> shim (`sh.exe` shipped with Git for Windows). If you invoke Rspack
> directly in PowerShell, set the variable explicitly first:
>
> ```powershell
> $env:BUILD_TARGET = 'tauri'; npx rsbuild build
> ```

## Where is my data?

The desktop shell pins the backend's `DATA_DIR`, `LOG_DIR` and matplotlib
cache to OS-conventional per-user directories (resolved via Tauri's
`app.path()` APIs from the `com.intel.geti` bundle identifier). These live
**outside** the install prefix, so reinstalls and upgrades preserve them
— same convention as Chrome and VSCode.

| Platform | Data                                           | Logs                            | Cache (matplotlib)                           |
| -------- | ---------------------------------------------- | ------------------------------- | -------------------------------------------- |
| macOS    | `~/Library/Application Support/com.intel.geti` | `~/Library/Logs/com.intel.geti` | `~/Library/Caches/com.intel.geti/matplotlib` |
| Windows  | `%APPDATA%\com.intel.geti`                     | `%APPDATA%\com.intel.geti\logs` | `%LOCALAPPDATA%\com.intel.geti\matplotlib`   |
| Linux    | `~/.local/share/com.intel.geti`                | `~/.local/state/com.intel.geti` | `~/.cache/com.intel.geti/matplotlib`         |

Set `DATA_DIR`, `LOG_DIR`, or `MPLCONFIGDIR` in the environment to override
any of them (the Rust shell only fills in what's missing).

## Cleanup / uninstall

The OS uninstaller / drag-to-trash only removes the app bundle. To wipe
**everything**:

1. **Per-user data** — delete the directories listed above. On macOS:
    ```sh
    rm -rf ~/Library/{Application\ Support,Logs,Caches}/com.intel.geti
    ```
2. **PyInstaller build output** (only if you built locally):
    ```sh
    rm -rf application/backend/dist application/backend/build
    ```
3. **Rust build artifacts**:
    ```sh
    rm -rf application/ui/src-tauri/target
    ```
4. **Frontend build output**:
    ```sh
    rm -rf application/ui/dist
    ```

## Verifying the platform split

Quick checks after changing anything under `src/platform/`:

```sh
# Web build should not contain any @tauri-apps references.
npm run build
grep -R "@tauri-apps" dist/ && echo "LEAK" || echo "clean"

# Tauri build must contain them.
npm run build:tauri
grep -R "tauri" dist/ >/dev/null && echo "ok" || echo "missing"
```
