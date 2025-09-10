// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import { defineConfig } from 'vitest/config';

const CI = !!process.env.CI;

export default defineConfig({
    plugins: [
        react(),
        svgr({
            svgrOptions: {
                svgo: false,
                exportType: 'named',
            },
            include: '**/*.svg',
        }),
    ],
    test: {
        environment: 'jsdom',

        coverage: {
            provider: 'v8',
            reporter: [CI ? 'json-summary' : 'text'],
            reportOnFailure: true,
            include: ['src/**/*.{ts,tsx}'],
        },

        // This is needed to use globals like describe or expect
        globals: true,
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: './src/setup-tests.ts',
        watch: false,
        server: {
            deps: {
                inline: [/@react-spectrum\/.*/, /@spectrum-icons\/.*/, /@adobe\/react-spectrum\/.*/],
            },
        },
    },
});
