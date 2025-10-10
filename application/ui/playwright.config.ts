// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from '@playwright/test';

const CI = !!process.env.CI;
// Docker mode: BASE_URL env var is set by Docker Compose (e.g., BASE_URL=http://frontend-e2e)
const USE_DOCKER = !!process.env.BASE_URL;
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
// Component tests only need frontend, E2E tests need backend + frontend
const IS_COMPONENT_TEST = process.argv.includes('--project=component');

const ACTION_TIMEOUT = 30000;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    /* Run tests in files in parallel */
    fullyParallel: true,
    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: CI,
    retries: 0,
    /* Opt out of parallel tests on CI. */
    workers: process.env.CI ? 1 : undefined,
    /* Test timeout */
    timeout: CI ? 120000 : 60000,
    /* Expect timeout */
    expect: {
        timeout: CI ? 10000 : 5000,
    },
    /* Reporter to use. See https://playwright.dev/docs/test-reporters */
    reporter: [[CI ? 'github' : 'list'], ['html', { open: 'never' }]],
    use: {
        baseURL: BASE_URL,
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
            devtools: !CI,
            // Additional browser args for WebRTC in headless mode
            args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--disable-web-security'],
        },
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        timezoneId: 'UTC',
        actionTimeout: ACTION_TIMEOUT,
        navigationTimeout: ACTION_TIMEOUT,
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'component',
            testDir: './tests',
            testIgnore: '**/e2e/**',
        },
        {
            name: 'e2e',
            testDir: './tests/e2e',
        },
    ],

    /* Run your local dev server before starting the tests */
    webServer: USE_DOCKER
        ? undefined // Docker Compose handles servers
        : IS_COMPONENT_TEST
          ? [
                // Component tests: only frontend server needed
                {
                    command: 'npm run start',
                    url: 'http://localhost:3000',
                    reuseExistingServer: !CI,
                    timeout: 120_000,
                },
            ]
          : [
                // E2E tests: start both backend and frontend
                {
                    command: './run.sh',
                    url: 'http://localhost:7860/health',
                    reuseExistingServer: !CI,
                    timeout: 120_000,
                    cwd: '../backend',
                    env: {
                        DATABASE_FILE: 'geti_tune_e2e.db',
                        SEED_DB: 'true',
                        DOWNLOAD_FILES: 'true',
                        PYTHONPATH: '.',
                        PYTHONUNBUFFERED: '1',
                    },
                },
                {
                    command: 'npm run start',
                    url: 'http://localhost:3000',
                    reuseExistingServer: !CI,
                    timeout: 120_000,
                },
            ],
});
