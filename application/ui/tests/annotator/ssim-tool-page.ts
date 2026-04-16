// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import type { Rect } from '../../src/shared/types';
import { clickAndMove, withRelative } from '../utils/mouse';

export class SSIMToolPage {
    constructor(private page: Page) {}

    async drawTemplate({ x, y, width, height }: Omit<Rect, 'type'>) {
        const relative = await withRelative(this.page);
        const startPoint = relative(x, y);
        const endPoint = relative(x + width, y + height);

        await clickAndMove(this.page, startPoint, endPoint);
    }

    getTool() {
        return this.page.getByRole('button', { name: 'ssim tool' });
    }

    async selectTool() {
        await this.getTool().click();
    }
}
