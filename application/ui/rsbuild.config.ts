// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginBabel } from '@rsbuild/plugin-babel';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars } = loadEnv({ prefixes: ['PUBLIC_'] });

// Platform target selection. When building for the Tauri desktop shell we
// prepend `.tauri.*` extensions so the bundler resolves platform-specific
// overrides (e.g. `foo.tauri.ts` wins over `foo.ts`). Files not shadowed by
// a `.tauri.*` twin resolve as usual. This keeps Tauri-specific code out of
// the web graph entirely, and removes the need for runtime `isTauri` checks.
const isTauriBuild = process.env.BUILD_TARGET === 'tauri';

// `TAURI_ENV_DEBUG` is set by the Tauri CLI: `tauri dev` / `start:desktop`
// propagate it as `true`, and `tauri build` sets it to `false`. We disable
// minification and emit inline JS source maps for debug desktop builds so
// stack traces are readable inside the embedded WebView.
const isTauriDebugBuild = isTauriBuild && process.env.TAURI_ENV_DEBUG === 'true';

const platformExtensions = isTauriBuild ? ['.tauri.tsx', '.tauri.ts', '.tauri.jsx', '.tauri.js', '.tauri.scss'] : [];
// `.scss` is appended unconditionally so extensionless SCSS imports (used
// to opt in to the platform-override mechanism, e.g. `import './foo'`)
// still resolve to `foo.scss` on the web build.
const styleExtensions = ['.scss'];

// API base URL injected into the bundle. Web builds resolve to '' (relative
// paths, served same-origin behind a reverse proxy). Tauri builds load the
// frontend from a custom protocol (`tauri://localhost`) that has no API, so
// fetches must point at the absolute backend URL — the sidecar always binds
// to localhost:7860. The web dev server reads the same value from .env.development.
const getPublicApiUrl = () => {
    if (publicVars['import.meta.env.PUBLIC_API_BASE_URL'] !== undefined) {
        return JSON.parse(publicVars['import.meta.env.PUBLIC_API_BASE_URL']);
    }

    if (isTauriBuild) {
        return 'http://localhost:7860';
    }

    return '';
};
const publicApiUrl = getPublicApiUrl();
const publicApiUrlJson = JSON.stringify(publicApiUrl);

export default defineConfig({
    plugins: [
        pluginReact(),

        // React Compiler
        pluginBabel({
            include: /\.(?:tsx)$/,
            babelLoaderOptions(opts) {
                opts.plugins?.unshift('babel-plugin-react-compiler');
            },
        }),

        pluginSass(),

        pluginSvgr({
            svgrOptions: {
                exportType: 'named',
            },
        }),
    ],
    output: {
        assetPrefix: process.env.ASSET_PREFIX,
        minify: isTauriDebugBuild ? false : undefined,
        sourceMap: isTauriDebugBuild
            ? {
                  js: 'inline-source-map',
                  css: false,
              }
            : undefined,
    },
    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_BASE_URL': publicApiUrlJson,
            'process.env.PUBLIC_API_BASE_URL': publicApiUrlJson,
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },
    html: {
        template: './public/index.html',
        title: 'Geti™',
        favicon: './src/assets/icons/favicon.ico',
        meta: {
            description:
                'Geti™ provides a "recipe" for every supported task type, which consolidates ' +
                'necessary information to build a model. Model templates are validated on ' +
                'various datasets and serve as a one-stop shop for obtaining the best models in general.',
        },
    },
    performance: {
        preload: {
            type: 'initial',
            include: [
                /roboto-flex-v30-latin-regular.*\.woff2$/,
                // The branded loading spinner is the LCP element on the initial
                // route (it's rendered by the root <Suspense> fallback while the
                // route chunk loads). Without a preload, the browser can't
                // discover its URL until ~2 MB of JS parses and React mounts,
                // pushing LCP to ~4 s. Preloading shrinks resourceLoadDelay
                // dramatically and lets the spinner paint near FCP.
                /intel-loading\..*\.webp$/,
            ],
        },
    },
    tools: {
        rspack: (config) => {
            // `resolve.extensions` is order-sensitive: the first match wins.
            // Rsbuild's defaults put `.ts` near the front, so a plain object
            // merge would let it shadow our `.tauri.ts` overrides. Prepend
            // explicitly and dedupe to keep the platform suffixes first.
            const existing = config.resolve?.extensions ?? [];
            const extensions = Array.from(new Set([...platformExtensions, ...existing, ...styleExtensions]));

            return {
                ...config,
                resolve: { ...config.resolve, extensions },
                watchOptions: { ...config.watchOptions, ignored: ['**/src-tauri/**'] },
            };
        },
    },
    server: {
        // Proxy backend calls to the FastAPI server so the web dev build can
        // use same-origin requests (empty PUBLIC_API_BASE_URL). This avoids
        // cross-origin CORS failures regardless of whether the app is opened
        // via localhost, 127.0.0.1 or a LAN IP. `ws: true` keeps SSE / job
        // streaming working through the proxy.
        proxy: isTauriBuild
            ? undefined
            : {
                  '/api': { target: 'http://localhost:7860', changeOrigin: true, ws: true },
                  '/health': { target: 'http://localhost:7860', changeOrigin: true },
              },
        headers: {
            'Cross-Origin-Embedder-Policy': 'credentialless',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cache-Control': 'public, max-age=31536000, immutable',
            'Content-Security-Policy':
                "default-src 'self'; " +
                "script-src 'self' 'unsafe-eval' blob:; " +
                "worker-src 'self' blob:; " +
                `connect-src 'self' ${publicApiUrl} data:; ` +
                `img-src 'self' ${publicApiUrl} data: blob:; ` +
                `media-src 'self' ${publicApiUrl} blob: data:; ` +
                "style-src 'self' 'unsafe-inline';",
        },
    },
});
