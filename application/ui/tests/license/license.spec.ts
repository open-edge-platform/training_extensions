// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from '../fixtures';

test.describe('License agreement', () => {
    test('shows the license screen and accepts the license', async ({ page, network }) => {
        let licenseAccepted = false;

        network.use(
            http.get('/api/system/info', ({ response }) => {
                return response(200).json({
                    license_accepted: licenseAccepted,
                    platform: 'linux',
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
            await expect(page.getByRole('link', { name: /DINOv2 License/i })).toBeVisible();
            await expect(page.getByRole('link', { name: /Apache License 2\.0/i })).toBeVisible();
        });

        await test.step('accepting the license hides the license screen', async () => {
            await page.getByRole('button', { name: /Accept and continue/i }).click();

            await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeHidden();
        });
    });

    test('shows Intel license on Windows platform', async ({ page, network }) => {
        network.use(
            http.get('/api/system/info', ({ response }) => {
                return response(200).json({
                    license_accepted: false,
                    platform: 'windows',
                });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('link', { name: /Intel Simplified Software License/i })).toBeVisible();
    });

    test('skips license screen when license is already accepted', async ({ page, network }) => {
        network.use(
            http.get('/api/system/info', ({ response }) => {
                return response(200).json({
                    license_accepted: true,
                    platform: 'linux',
                });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('heading', { name: /License Agreement/i })).toBeHidden();
    });
});
