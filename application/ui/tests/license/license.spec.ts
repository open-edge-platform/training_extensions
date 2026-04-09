// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from '../fixtures';

test.describe('License agreement', () => {
    test('shows the license screen and accepts the license', async ({ page, network }) => {
        let licenseAccepted = false;

        network.use(
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                    license_accepted: licenseAccepted,
                });
            }),
            http.post('/api/license/accept', ({ response }) => {
                licenseAccepted = true;

                return response(200).json({ license_accepted: true });
            })
        );

        await test.step('license screen is shown when license is not accepted', async () => {
            await page.goto('/');

            await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeVisible();
            await expect(page.getByRole('link', { name: /DINOv3 License/i })).toBeVisible();
        });

        await test.step('accepting the license hides the license screen', async () => {
            await page.getByRole('button', { name: /Accept and continue/i }).click();

            await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeHidden();
        });
    });

    test('skips license screen when license is already accepted', async ({ page, network }) => {
        network.use(
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                    license_accepted: true,
                });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeHidden();
    });
});
