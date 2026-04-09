// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from '../fixtures';

test.describe('License Agreement', () => {
    test('shows license dialog and proceeds after acceptance', async ({ page, network }) => {
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

        await page.goto('/');

        await test.step('license dialog is shown when license is not accepted', async () => {
            await expect(page.getByRole('heading', { name: 'License Agreement' })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Accept' })).toBeDisabled();
        });

        await test.step('accepts license and app loads', async () => {
            await page.getByRole('checkbox').check();
            await expect(page.getByRole('button', { name: 'Accept' })).toBeEnabled();
            await page.getByRole('button', { name: 'Accept' }).click();

            await expect(page.getByRole('heading', { name: 'License Agreement' })).toBeHidden();
        });
    });

    test('skips license dialog when license is already accepted', async ({ page, network }) => {
        network.use(
            http.get('/health', ({ response }) => {
                return response(200).json({
                    status: 'ok',
                    license_accepted: true,
                });
            })
        );

        await page.goto('/');

        await expect(page.getByRole('heading', { name: 'License Agreement' })).toBeHidden();
    });
});
