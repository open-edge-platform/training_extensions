// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from '@playwright/test';

const CI = !!process.env.CI;

const ACTION_TIMEOUT = 30000;
const SERVER_TIMEOUT = 60000;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    testDir: './tests/e2e',
    fullyParallel: false,
    forbidOnly: CI,
    retries: process.env.CI ? 2 : 0,
    workers: 1,
    timeout: CI ? 120000 : 60000,
    expect: {
        timeout: CI ? 10000 : 5000,
    },
    reporter: [[CI ? 'github' : 'list'], ['html', { open: 'never' }]],
    use: {
        baseURL: 'http://localhost:3000',
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        actionTimeout: ACTION_TIMEOUT,
        navigationTimeout: ACTION_TIMEOUT,
    },

    projects: [
        {
            name: 'E2E',
            use: {
                ...devices['Desktop Chrome'],
                headless: CI,
                viewport: { width: 1280, height: 720 },
            },
        },
    ],

    webServer: [
        {
            command: 'cd ../backend && rm -f data/geti_tune.db && ./run.sh',
            name: 'backend',
            url: 'http://localhost:7860/health',
            reuseExistingServer: !CI,
            timeout: SERVER_TIMEOUT,
        },
        {
            command: CI ? 'npx serve -s dist -p 3000' : 'npm run start',
            name: 'frontend',
            url: 'http://localhost:3000',
            reuseExistingServer: !CI,
            timeout: ACTION_TIMEOUT,
        },
    ],
});
