// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { getZoomLevel } from './zoom';

type Relative = (x: number, y: number) => { x: number; y: number };

export const withRelative = async (page: Page): Promise<Relative> => {
    const zoom = await getZoomLevel(page);
    const rect = (await page.getByTestId('annotations-canvas-tools').boundingBox()) ?? { x: 0, y: 0 };

    const relative = (x: number, y: number): ReturnType<Relative> => {
        return { x: x * zoom + rect.x, y: y * zoom + rect.y };
    };

    return relative;
};

export const clickAndMove = async (
    page: Page,
    startPoint: { x: number; y: number },
    endPoint: { x: number; y: number },
    options?: Parameters<Page['mouse']['down']>[0]
) => {
    await page.mouse.move(startPoint.x, startPoint.y);
    await page.mouse.down(options);
    await page.mouse.move(endPoint.x, endPoint.y);
    await page.mouse.up(options);
};
