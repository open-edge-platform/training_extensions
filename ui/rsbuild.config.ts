// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginBabel } from '@rsbuild/plugin-babel';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars } = loadEnv({ prefixes: ['PUBLIC_'] });

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

    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_BASE_URL':
                publicVars['import.meta.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:7860"',
            'process.env.PUBLIC_API_BASE_URL':
                publicVars['process.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:7860"',
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },
    html: {
        title: 'Geti Tune',
        favicon: './src/assets/icons/build-icon.svg',
    },
    tools: {
        rspack: {
            watchOptions: {
                ignored: ['**/src-tauri/**'],
            },
        },
    },
});
