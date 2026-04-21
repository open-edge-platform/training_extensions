// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect } from '@playwright/test';
import { getMockedVideoFrame } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { AnnotationDTO } from '../../src/constants/shared-types';
import { http, test } from '../fixtures';
import { candyPngBuffer, redLabel } from './annotator-fixtures';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel],
    },
});

const mockAnnotation: AnnotationDTO = {
    shape: {
        type: 'rectangle',
        x: 100,
        y: 100,
        width: 220,
        height: 180,
    },
    labels: [{ id: redLabel.id }],
};

const mockVideoFrame = getMockedVideoFrame({
    id: 'video-1',
    name: 'video-1.mp4',
    frame_number: 0,
    frame_count: 3600,
    frame_stride: 60,
    fps: 60,
    duration: 1000 * 60 * 60,
    width: 960,
    height: 540,
});

const totalFrames = mockVideoFrame.frame_count - 1;

const videoGalleryItem = {
    ...mockVideoFrame,
    video_id: 'video-parent-1',
    frame_index: mockVideoFrame.frame_number,
};

type SubmittedFrameRequest = {
    frameIndex: number;
    annotations: AnnotationDTO[];
};

const dirname = path.dirname(fileURLToPath(import.meta.url));
const videoFilePath = path.resolve(dirname, '../assets/fish_60.mp4');

test.describe.only('Annotator video player', () => {
    let frameAnnotations: Record<number, AnnotationDTO[]>;
    let submittedFrameRequests: SubmittedFrameRequest[];

    test.beforeEach(async ({ network }) => {
        frameAnnotations = {
            0: [mockAnnotation],
            1: [],
            2: [],
            3: [],
            4: [],
        };
        submittedFrameRequests = [];

        const videoFile = fs.readFileSync(videoFilePath);

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: [videoGalleryItem],
                    pagination: { offset: 0, limit: 20, count: 1, total: 1 },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', ({ query }) => {
                if (query.has('frame_index')) {
                    return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                        headers: {
                            'Content-Type': 'image/png',
                        },
                    });
                }

                return HttpResponse.arrayBuffer(
                    videoFile.buffer.slice(videoFile.byteOffset, videoFile.byteOffset + videoFile.byteLength),
                    {
                        headers: { 'Content-Type': 'video/mp4', 'Content-Length': videoFile.byteLength.toString() },
                    }
                );
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ request }) => {
                const frameIndex = Number(new URL(request.url).searchParams.get('frame_index') ?? '0');

                return HttpResponse.json({
                    annotations: frameAnnotations[frameIndex] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', () => {
                return HttpResponse.json(
                    Object.entries(frameAnnotations).map(([frameIndex, annotations]) => ({
                        media_id: mockVideoFrame.id,
                        frame_index: Number(frameIndex),
                        annotation_data: {
                            annotations,
                            subset: 'training' as const,
                            user_reviewed: true,
                        },
                    }))
                );
            }),
            http.post('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ request }) => {
                const url = new URL(request.url);
                const frameIndex = Number(url.searchParams.get('frame_index') ?? '0');
                const body = (await request.json()) as { annotations: AnnotationDTO[] };

                frameAnnotations[frameIndex] = body.annotations;
                submittedFrameRequests.push({ frameIndex, annotations: body.annotations });

                return HttpResponse.json(
                    { annotations: body.annotations, user_reviewed: true, subset: 'training' },
                    { status: 201 }
                );
            })
        );
    });

    test('loads video controls and timeline for a video item', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await expect(videoPage.getPlayButton()).toBeVisible();
        await expect(videoPage.getPreviousFrameButton()).toBeDisabled();
        await expect(videoPage.getNextFrameButton()).toBeEnabled();
        await expect(videoPage.getVideoTimeline()).toBeVisible();
        await expect(videoPage.getVideoDuration()).toBeVisible();
    });

    test('toggles play and pause for video playback', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await videoPage.play();
        await expect(videoPage.getPauseButton()).toBeVisible();

        await videoPage.pauseVideo();
        await expect(videoPage.getPlayButton()).toBeVisible();
    });

    test('navigates frames and updates frame annotations', async ({ annotatorPage, videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await videoPage.expandToolbar();
        await videoPage.expectCurrentFrame(0, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);

        await videoPage.previousFrame();
        await videoPage.expectCurrentFrame(0, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
    });

    test('adds annotation on video frame and submits', async ({ annotatorPage, boundingBoxTool, videoPage }) => {
        await test.step('Opens annotator', async () => {
            await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);
        });

        await test.step('Selects frame to annotate', async () => {
            await videoPage.expandToolbar();
            await videoPage.nextFrame();
            await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
        });

        await test.step('Draws an annotation on video frame', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 260, y: 140, width: 180, height: 120 });

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
            await expect(videoPage.getSubmitButton()).toBeEnabled();
        });

        await test.step('Submits annotation on video frame', async () => {
            await videoPage.getSubmitButton().click();

            expect(submittedFrameRequests).toHaveLength(1);
            expect(submittedFrameRequests[0].frameIndex).toBe(mockVideoFrame.frame_stride);
            expect(submittedFrameRequests[0].annotations).toHaveLength(1);
            expect(submittedFrameRequests[0].annotations[0].shape.type).toBe('rectangle');
        });

        await test.step('Navigates to the next frame automatically', async () => {
            await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride * 2, totalFrames);
        });
    });

    test('disables next frame button at last frame boundary, disable previous frame at first frame boundary', async ({
        videoPage,
    }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);
        await videoPage.expandToolbar();

        await expect(videoPage.getPreviousFrameButton()).toBeDisabled();

        const lastFrame = mockVideoFrame.frame_count - mockVideoFrame.frame_stride;
        await videoPage.selectFrame(lastFrame);

        await videoPage.expectCurrentFrame(lastFrame, totalFrames);
        await expect(videoPage.getNextFrameButton()).toBeDisabled();
        await expect(videoPage.getPreviousFrameButton()).toBeEnabled();
    });

    test('toggles frame mode and disables frame mode while playing', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);
        await videoPage.expandToolbar();

        // In 1/1 mode, frame step follows default frame_stride/fps behavior (here, +2).
        await expect(videoPage.getFrameModeIndicator()).toHaveText('1/1');
        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
        await videoPage.previousFrame();
        await videoPage.expectCurrentFrame(0, totalFrames);

        // In ALL mode, frame step is 1 frame.
        await videoPage.toggleFrameMode();
        await expect(videoPage.getFrameModeIndicator()).toHaveText('ALL');
        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(1, totalFrames);

        await videoPage.play();
        await expect(videoPage.getPauseButton()).toBeVisible();
        await expect(videoPage.getToggleFrameModeButton()).toBeDisabled();

        await videoPage.pauseVideo();
        await expect(videoPage.getPlayButton()).toBeVisible();
        await expect(videoPage.getToggleFrameModeButton()).toBeEnabled();

        await videoPage.toggleFrameMode();
        await expect(videoPage.getFrameModeIndicator()).toHaveText('1/1');
    });
});
