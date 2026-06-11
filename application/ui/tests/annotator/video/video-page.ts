// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, Locator, Page } from '@playwright/test';

export class VideoPage {
    constructor(private readonly page: Page) {}

    async openVideoFromDataset(projectId: string, videoName: string) {
        await this.page.goto(`/projects/${projectId}/dataset`);
        await this.page.getByRole('img', { name: videoName }).dblclick();
    }

    getPlayButton(): Locator {
        return this.page.getByRole('button', { name: 'Play video' });
    }

    getPauseButton(): Locator {
        return this.page.getByRole('button', { name: 'Pause video' });
    }

    getPreviousFrameButton(): Locator {
        return this.page.getByRole('button', { name: 'Go to previous frame' });
    }

    getNextFrameButton(): Locator {
        return this.page.getByRole('button', { name: 'Go to next frame' });
    }

    getExpandToolbarButton(): Locator {
        return this.page.getByRole('button', { name: 'Expand toolbar' });
    }

    getVideoTimeline(): Locator {
        return this.page.getByLabel('Video timeline');
    }

    getVideoDuration(): Locator {
        return this.page.getByLabel('Video duration');
    }

    getSubmitButton(): Locator {
        return this.page.getByRole('button', { name: 'Submit' });
    }

    getMuteButton(): Locator {
        return this.page.getByRole('button', { name: 'Mute audio' });
    }

    getUnmuteButton(): Locator {
        return this.page.getByRole('button', { name: 'Unmute audio' });
    }

    getToggleFrameModeButton(): Locator {
        return this.page.getByRole('button', { name: 'Toggle frame mode' });
    }

    getFrameModeIndicator(): Locator {
        return this.page.getByTestId('frame-mode-indicator-id');
    }

    async expandToolbar() {
        await this.getExpandToolbarButton().click();
    }

    async play() {
        await this.getPlayButton().click();
    }

    async pauseVideo() {
        await this.getPauseButton().click();
    }

    async nextFrame() {
        await this.getNextFrameButton().click();
    }

    async previousFrame() {
        await this.getPreviousFrameButton().click();
    }

    async toggleFrameMode() {
        await this.getToggleFrameModeButton().click();
    }

    async mute() {
        await this.getMuteButton().click();
    }

    async unmute() {
        await this.getUnmuteButton().click();
    }

    async expectCurrentFrame(frame: number, totalFrames: number) {
        await expect(this.page.getByText(`Current frame: ${frame} / Total frames: ${totalFrames}`)).toBeVisible();
    }

    getLabelSegment(frameNumber: number, label: string) {
        return this.page.getByRole('gridcell', {
            name: `Label ${label} in frame number ${frameNumber}`,
            exact: true,
        });
    }

    async selectFrame(frameNumber: number) {
        await this.page
            .getByRole('gridcell', { name: new RegExp(`in frame number ${frameNumber}`, 'i') })
            .first()
            .click();
    }
}
