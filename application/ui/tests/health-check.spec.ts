// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from './fixtures';

test.describe('Health Check', () => {
    test('Shows loading while retrying and renders app once healthy', async ({ page, network }) => {
        let healthCheckAttempts = 0;

        network.use(
            http.get('/health', ({ response }) => {
                healthCheckAttempts += 1;

                // Fail the initial call and both retries, succeed on the 3rd attempt
                if (healthCheckAttempts < 5) {
                    // @ts-expect-error We want to mock behavior when server is not ready yet
                    return response(500).json({});
                }

                return response(200).json({ status: 'ok' });
            })
        );

        await page.goto('/');

        // Loading is shown during retries
        await expect(page.getByRole('progressbar')).toBeVisible();

        // With a 5s retry delay, the 3rd attempt succeeds after ~10s, so extend the timeout.
        await expect(page.getByRole('progressbar')).toBeHidden({ timeout: 20_000 });
        await expect(page.getByRole('heading', { name: 'Server Error' })).toBeHidden();
    });

    test('Shows error message only after all retries are exhausted', async ({ page, network }) => {
        network.use(
            http.get('/health', ({ response }) => {
                // @ts-expect-error Simulate server error
                return response(500).json({});
            })
        );

        await page.goto('/');

        // Loading is shown while retrying
        await expect(page.getByRole('progressbar')).toBeVisible();

        // Error shown only after all 6 attempts (initial + 5 retries) fail.
        // With a 5s retry delay, this can take ~25s, so we extend the timeout.
        await expect(page.getByRole('heading', { name: 'Server Error' })).toBeVisible({ timeout: 40_000 });
        await expect(page.getByRole('button', { name: 'Refresh' })).toBeVisible();
        await expect(page.getByRole('progressbar')).toBeHidden();
    });
});
