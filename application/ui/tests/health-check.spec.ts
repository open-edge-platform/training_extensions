// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from './fixtures';

test.describe('Health Check', () => {
    test('Shows loading when health check is pending', async ({ page, network }) => {
        let healthCheckAttempts = 0;

        network.use(
            http.get('/health', ({ response }) => {
                healthCheckAttempts += 1;

                if (healthCheckAttempts < 3) {
                    // @ts-expect-error We want to mock behavior when server is not ready yet
                    return response(500).json({});
                }

                return response(200).json({ status: 'ok' });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('progressbar')).toBeVisible();

        await expect(page.getByRole('progressbar')).toBeHidden();
    });
});
