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
import { downloadFile } from '@/platform/download-file';
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
src/platform/
  download-file.ts         ← web (default) implementation
  download-file.tauri.ts   ← tauri override (fetches via webview and triggers a blob anchor download to Downloads/)
  …                        ← future capabilities follow the same pair
```

Rules of thumb when adding a platform-specific behaviour:

1. **Plain file is the default.** It runs in both web and Tauri _unless_
   shadowed by a `.tauri.*` twin.
2. **`.tauri.*` files may import `@tauri-apps/*`.** Other source files may
   not — this is enforced by the `no-restricted-imports` rule in
   [`../eslint.config.js`](../eslint.config.js).
3. **Tauri-only features:** ship a no-op/null-returning module as the default
   and the real implementation in `.tauri.tsx`. Consumers render/call
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
  `brew install just`, `cargo install just`, or your package manager).
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

From `application/ui/src-tauri`:

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

The symlinks are gitignored. The Rust shell spawns the executable that
lives next to the bundled app at runtime — see `spawn_backend()` in
[`src/main.rs`](./src/main.rs).

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
    npx tauri build
    ```
    Artifacts land in `src-tauri/target/release/bundle/`.

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
    npx tauri build
    ```
    Artifacts land in `src-tauri/target/release/bundle/`.

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
    npx tauri build
    ```
    Artifacts land in `src-tauri\target\release\bundle\`.

> The `start:tauri` and `build:tauri` npm scripts set `BUILD_TARGET=tauri`
> using POSIX shell syntax (`BUILD_TARGET=tauri rsbuild …`). On native
> Windows shells this works because npm runs scripts through a POSIX-ish
> shim (`sh.exe` shipped with Git for Windows). If you invoke Rspack
> directly in PowerShell, set the variable explicitly first:
>
> ```powershell
> $env:BUILD_TARGET = 'tauri'; npx rsbuild build
> ```

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
