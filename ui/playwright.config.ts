import { defineConfig, devices } from '@playwright/test';

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    testDir: './tests',
    /* Run tests in files in parallel */
    fullyParallel: true,
    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: !!process.env.CI,
    /* Retry on CI only */
    retries: process.env.CI ? 2 : 0,
    /* Opt out of parallel tests on CI. */
    workers: process.env.CI ? 1 : undefined,
    /* Reporter to use. See https://playwright.dev/docs/test-reporters */
    reporter: 'html',
    use: {
        baseURL: 'http://localhost:3000',
        trace: process.env.CI ? 'on-first-retry' : 'on',
        video: process.env.CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
            devtools: true,
        },
        timezoneId: 'UTC',
        actionTimeout: process.env.CI ? 10000 : 5000,
        navigationTimeout: process.env.CI ? 10000 : 5000,
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'Component tests',
            use: { ...devices['Desktop Chrome'] },
        },
    ],

    /* Run your local dev server before starting the tests */
    webServer: {
        command: process.env.CI ? 'npx serve -s dist -p 3000' : 'npm run dev',
        name: 'client',
        url: 'http://localhost:3000',
        reuseExistingServer: !process.env.CI,
    },
});
