// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export const getZoomLevel = async (page: Page): Promise<number> => {
    return Number(await page.getByLabel('Zoom level').getAttribute('data-value'));
};
