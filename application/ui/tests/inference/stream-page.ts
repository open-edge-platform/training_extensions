// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export class StreamPage {
    constructor(private page: Page) {}

    getStartStreamButton() {
        return this.page.getByRole('button', { name: 'Start stream' });
    }

    async startStream() {
        await this.getStartStreamButton().click();
    }

    async isConnected() {
        return await this.page.getByLabel('Connected').isVisible();
    }

    async stopStream() {
        await this.page.getByLabel('Stop stream').click();
    }
}
