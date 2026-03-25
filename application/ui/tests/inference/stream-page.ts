// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export class StreamPage {
    constructor(private page: Page) {}

    async startStream() {
        await this.page.getByLabel('Start stream').click();
    }

    async isConnected() {
        return await this.page.getByLabel('Connected').isVisible();
    }

    async stopStream() {
        await this.page.getByLabel('Stop stream').click();
    }
}
