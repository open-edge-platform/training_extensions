// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from '@playwright/test';

const CI = !!process.env.CI;

const ACTION_TIMEOUT = 30000;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    /* Run tests in files in parallel */
    fullyParallel: true,
    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: CI,
    /* Retry on CI only */
    retries: process.env.CI ? 2 : 0,
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
        baseURL: 'http://localhost:3000',
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
            devtools: true,
        },
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
            use: {
                ...devices['Desktop Chrome'],
                headless: true,
                viewport: { width: 1280, height: 720 },
            },
        },
        {
            name: 'e2e',
            testDir: './tests/e2e',
            use: {
                ...devices['Desktop Chrome'],
                headless: CI,
                viewport: { width: 1280, height: 720 },
            },
        },
    ],

    /* Run your local dev server before starting the tests */
    webServer: !process.env.ENABLE_BACKEND
        ? {
              command: CI ? 'npx serve -s dist -p 3000' : 'npm run start',
              url: 'http://localhost:3000',
              reuseExistingServer: true,
              timeout: ACTION_TIMEOUT,
          }
        : undefined,
});
