// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { NetworkFixture } from '@msw/playwright';
import { getMockedLabel } from 'mocks/mock-labels';
import {
    getMockedMediaImage,
    getMockedVideo,
    getMockedVideoFrame,
    getMultipleMockedMediaImage,
} from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { v4 as uuid } from 'uuid';

import { SchemaProjectView } from '../../src/api/openapi-spec';
import { AnnotationDTO, MediaDTO } from '../../src/constants/shared-types';
import { expect, http, test } from '../fixtures';

const mockedItems = getMultipleMockedMediaImage(20, '1');
const mockedItems2 = getMultipleMockedMediaImage(20, '2');
const mockedItems3 = getMultipleMockedMediaImage(20, '3');
const totalElements = mockedItems.length + mockedItems2.length + mockedItems3.length;

const dirname = path.dirname(fileURLToPath(import.meta.url));
const sampleImagePath = path.resolve(dirname, '../assets/candy-thumbnail.png');
const sampleImageBuffer = fs.readFileSync(sampleImagePath);

const sampleVideoPath = path.resolve(dirname, '../assets/fish_60.mp4');

test.describe('Dataset', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', ({}) => {
                return HttpResponse.arrayBuffer(sampleImageBuffer.buffer, {
                    headers: { 'Content-Type': 'image/png' },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media', ({ query }) => {
                const offset = Number(query.get('offset') ?? 0);
                const limit = Number(query.get('limit'));
                const items = offset === 0 ? mockedItems : offset === 20 ? mockedItems2 : mockedItems3;

                return HttpResponse.json({
                    items,
                    pagination: {
                        offset,
                        limit,
                        count: items.length,
                        total: totalElements,
                    },
                });
            })
        );
    });

    test('list items', async ({ page }) => {
        await page.goto('projects/id-1/dataset');
        const loadedItems = 40;

        await expect(page.getByText(`${loadedItems} images`)).toBeVisible();

        await page.getByLabel('select all').click();

        await expect(page.getByText(`${loadedItems} selected`)).toBeVisible();
    });

    test('select multiple images', async ({ page }) => {
        const selectedElements = 5;

        await page.goto('projects/id-1/dataset');

        await expect(page.getByText('40 images')).toBeVisible();

        const listbox = page.getByRole('listbox', { name: 'data-collection-grid' });
        const options = listbox.getByRole('option');

        for (let i = 0; i < selectedElements; i++) {
            await options.nth(i).click();
        }

        await expect(page.getByText(`${selectedElements} selected`)).toBeVisible();
    });

    test('loads additional items when scrolling to the end of the container', async ({ page }) => {
        await page.goto('projects/id-1/dataset');

        await expect(page.getByText('40 images')).toBeVisible();

        await page.getByRole('listbox', { name: 'data-collection-grid' }).press('End');

        await expect(page.getByText(`${totalElements} images`)).toBeVisible();
    });

    test('selected media item is saved in the URL', async ({ page, annotatorPage }) => {
        const [firstElement] = mockedItems;
        await page.goto('projects/id-1/dataset');
        await expect(annotatorPage.getAnnotationsList()).not.toBeInViewport();

        await page.getByRole('img', { name: firstElement.name, exact: true }).dblclick();

        await expect(annotatorPage.getAnnotationsList()).toBeInViewport();

        expect(page.url()).toContain(`/dataset/${firstElement.id}`);
    });

    test('upload shows start, progress and finish toasts', async ({ page, network }) => {
        let uploadRequestCount = 0;
        const firstBatchCount = 10;
        const totalFiles = firstBatchCount + 1;

        network.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                uploadRequestCount += 1;
                const isLastUploadRequest = uploadRequestCount === totalFiles;

                // Small delay to test the "in progress" toast or else we would only see start and finish toasts
                await new Promise((resolve) => setTimeout(resolve, isLastUploadRequest ? 250 : 30));

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), {
                    status: 201,
                });
            })
        );

        await page.goto('projects/id-1/dataset');

        await page.getByLabel('Upload media files').setInputFiles(
            Array.from({ length: totalFiles }, (_, index) => ({
                name: `upload-${index + 1}.png`,
                mimeType: 'image/png',
                buffer: sampleImageBuffer,
            }))
        );

        await expect(page.getByRole('button', { name: 'Upload media' })).toBeDisabled();

        await expect(
            page.getByText(`Uploading ${totalFiles} item(s)... (${firstBatchCount} succeeded, 0 failed)`)
        ).toBeVisible();

        await expect(page.getByText(`Uploaded ${totalFiles} item(s)`)).toBeVisible();
    });

    test.describe('Bulk labelling', () => {
        const mockedImages = [getMockedMediaImage({ id: uuid() }), getMockedMediaImage({ id: uuid() })];
        const mockedVideo = getMockedVideo({ id: uuid() });
        const mockedMedia = [...mockedImages, mockedVideo];
        const mockedLabels = [
            getMockedLabel({
                id: 'id-cat',
                name: 'cat',
            }),
            getMockedLabel({
                id: 'id-dog',
                name: 'dog',
            }),
        ];
        const sampleVideoBuffer = fs.readFileSync(sampleVideoPath);

        const filesToUpload = [
            ...mockedImages.map((_, idx) => ({
                name: `upload-${idx + 1}.png`,
                mimeType: 'image/png',
                buffer: sampleImageBuffer,
            })),
            { name: 'upload-video.mp4', mimeType: 'video/mp4', buffer: sampleVideoBuffer },
        ];

        const mockNetwork = (network: NetworkFixture, project: SchemaProjectView) => {
            const createAnnotationPayloads: [string, AnnotationDTO[]][] = [];
            let getMediaCount = 0;

            network.use(
                http.post('/api/projects/{project_id}/dataset/media', async () => {
                    const media = mockedMedia[getMediaCount];

                    getMediaCount++;

                    return HttpResponse.json(media, {
                        status: 201,
                    });
                }),
                http.get('/api/projects/{project_id}', async () => {
                    return HttpResponse.json(project);
                }),
                http.post(
                    '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                    async ({ request, params }) => {
                        const payload = await request.json();

                        createAnnotationPayloads.push([params.media_id, payload.annotations]);

                        return HttpResponse.json({
                            annotations: payload.annotations,
                            user_reviewed: true,
                        });
                    }
                )
            );

            return createAnnotationPayloads;
        };

        test('Single label: bulk labelling is only disabled for classification task and only for images', async ({
            network,
            page,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: true,
                        labels: mockedLabels,
                    },
                })
            );

            await page.goto('projects/id-1/dataset');

            await page.getByLabel('Upload media files').setInputFiles(filesToUpload);

            await expect(page.getByRole('heading', { name: 'Label assignment' })).toBeVisible();

            await page.getByRole('checkbox', { name: `Select ${mockedLabels[0].name}` }).click();

            await page.getByRole('button', { name: 'Continue' }).click();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });

        test('Multi label: bulk labelling is only disabled for classification task and only for images', async ({
            network,
            page,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: false,
                        labels: mockedLabels,
                    },
                })
            );

            await page.goto('projects/id-1/dataset');

            await page.getByLabel('Upload media files').setInputFiles(filesToUpload);

            await expect(page.getByRole('heading', { name: 'Label assignment' })).toBeVisible();

            await page.getByRole('checkbox', { name: `Select ${mockedLabels[0].name}` }).click();
            await page.getByRole('checkbox', { name: `Select ${mockedLabels[1].name}` }).click();

            await page.getByRole('button', { name: 'Continue' }).click();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });
    });
});
