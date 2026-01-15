// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export class StreamPage {
    constructor(private page: Page) {}

    async startStream() {
        await this.page.getByRole('button', { name: 'Start Stream' }).click();
    }

    async isConnected() {
        return await this.page.getByLabel('Connected').isVisible();
    }

    async stopStream() {
        await this.page.getByRole('button', { name: 'Stop' }).click();
    }
}
