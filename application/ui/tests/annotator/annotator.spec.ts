// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { AnnotationDTO, DatasetSubset, PredictionDTO } from '../../src/constants/shared-types';
import { Polygon } from '../../src/shared/types';
import { http, test } from '../fixtures';
import { blueLabel, candyBinaryHandler, redLabel } from './annotator-fixtures';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel, blueLabel],
    },
});

test.describe('Annotator', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            candyBinaryHandler,
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );
    });

    test('Add and change annotations labels', async ({ page, boundingBoxTool, annotatorPage }) => {
        await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });

        await test.step('Change annotation label by clicking label badge', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            await page.getByLabel('annotation rect').nth(1).click();

            await expect(page.getByRole('button', { name: `Label ${redLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(1);
            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );
        });

        await test.step('Draw a second annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 300, y: 200, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(2);
        });

        await test.step('Change second annotation to red label', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            await page.getByLabel('annotation rect').nth(3).click();
            await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });

        await test.step('Verify both annotations have correct labels', async () => {
            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(1);
            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });
    });

    test('change multiple labels at once', async ({ page, boundingBoxTool, annotatorPage }) => {
        await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

        const annotations = [
            { x: 100, y: 100, width: 150, height: 150 },
            { x: 300, y: 200, width: 150, height: 150 },
            { x: 600, y: 300, width: 150, height: 150 },
        ];

        await test.step('Draw annotations', async () => {
            await boundingBoxTool.selectTool();

            for await (const annotation of annotations) {
                await boundingBoxTool.drawBoundingBox(annotation);
            }

            expect(await page.getByLabel(`label ${redLabel.name} background`).count()).toBe(annotations.length);
        });

        await test.step('Remove labels', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            const labels = page.getByLabel('Remove red-label');

            await labels.nth(0).click();
            await labels.nth(1).click();
        });

        await test.step('Change selected annotations label using label badge', async () => {
            const container = page.getByLabel('annotation rect');

            await container.nth(5).click();
            await container.nth(4).click({ modifiers: ['Shift'] });
            await container.nth(3).click({ modifiers: ['Shift'] });

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            expect(await page.getByLabel(`label ${blueLabel.name} background`).count()).toBe(annotations.length);
        });
    });

    test.describe('Handles empty label', () => {
        test('label assignment', async ({ page, boundingBoxTool, annotatorPage }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            const annotations = [
                { x: 100, y: 100, width: 150, height: 150 },
                { x: 300, y: 200, width: 150, height: 150 },
                { x: 600, y: 300, width: 150, height: 150 },
            ];

            await test.step('Draw annotations', async () => {
                await boundingBoxTool.selectTool();

                for await (const annotation of annotations) {
                    await boundingBoxTool.drawBoundingBox(annotation);
                }

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(annotations.length);
            });

            await test.step('Assigning "No object" removes other annotations', async () => {
                await page.getByLabel('Label No object').click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(0);
                await expect(page.getByLabel(`label No object background`)).toHaveCount(1);
            });

            await test.step('Drawing new annotation removes "No object" annotation', async () => {
                await boundingBoxTool.selectTool();

                await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
                await expect(page.getByLabel(`label No object background`)).toHaveCount(0);
            });
        });

        test('renders "No object" when server returns empty annotations list', async ({ page, annotatorPage }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await expect(page.getByLabel(`label No object background`)).toHaveCount(1);
        });
    });

    test('Tool selection persists across media items', async ({ page, polygonTool, annotatorPage, network }) => {
        const smallPolygon: Polygon = {
            type: 'polygon',
            points: [
                { x: 100, y: 100 },
                { x: 150, y: 100 },
                { x: 150, y: 150 },
                { x: 100, y: 150 },
            ],
        };

        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];
        const mockedSegmentationProject = getMockedProject({
            id: '123e4567-e89b-12d3-a456-426614174000',
            task: {
                exclusive_labels: true,
                task_type: 'instance_segmentation',
                labels: [redLabel, blueLabel],
            },
        });

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedSegmentationProject);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedSegmentationProject.id, 'media-1');

        await test.step('Select polygon tool on first media item', async () => {
            await polygonTool.selectPolygonTool();
        });

        await test.step('Navigate to second media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
        });

        await test.step('Verify polygon tool persisted by drawing a polygon', async () => {
            // If polygon tool persisted, we should be able to draw immediately without reselecting
            await polygonTool.drawPolygon(smallPolygon);

            expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(1);
        });

        await test.step('Navigate back to first media item', async () => {
            await annotatorPage.selectMediaItem('item-1.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
        });

        await test.step('Verify polygon tool still works after navigating back', async () => {
            // Draw another polygon to verify tool is still active
            await polygonTool.drawPolygon(smallPolygon);

            expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(1);
        });

        await test.step('Verify tool resets when switching modes', async () => {
            // Select SAM tool because polygon is the default tool for segmentation projects
            await page.getByRole('button', { name: 'sam tool' }).click();

            await annotatorPage.openPredictionMode();

            await expect(page.getByTestId('primary-toolbar-id')).toBeHidden();

            await annotatorPage.openAnnotationMode();

            await expect(page.getByTestId('primary-toolbar-id')).toBeVisible();

            // Verify polygon tool is active by drawing a polygon without manually selecting it
            await polygonTool.drawPolygon(smallPolygon);
            expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(2);
        });
    });

    test('Annotations reset correctly when switching media items', async ({ annotatorPage, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-reset-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-reset-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        const mediaAnnotations: Record<string, AnnotationDTO[]> = {
            'media-reset-1': [
                {
                    shape: {
                        type: 'rectangle',
                        x: 80,
                        y: 120,
                        width: 140,
                        height: 120,
                    },
                    labels: [{ id: redLabel.id }],
                },
            ],
            'media-reset-2': [],
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ params }) => {
                return HttpResponse.json({
                    annotations: mediaAnnotations[params.media_id] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-reset-1');

        await test.step('Check first media annotations', async () => {
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
        });

        await test.step('Switching to media 2 clears media 1 annotations', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
        });

        await test.step('Switching back restores media 1 annotations', async () => {
            await annotatorPage.selectMediaItem('item-1.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
        });
    });

    test('Selected annotations reset when switching media items', async ({ page, annotatorPage, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-selection-reset-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-selection-reset-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        const mediaAnnotations: Record<string, AnnotationDTO[]> = {
            'media-selection-reset-1': [
                {
                    shape: {
                        type: 'rectangle',
                        x: 80,
                        y: 120,
                        width: 140,
                        height: 120,
                    },
                    labels: [{ id: redLabel.id }],
                },
            ],
            'media-selection-reset-2': [],
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ params }) => {
                return HttpResponse.json({
                    annotations: mediaAnnotations[params.media_id] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-selection-reset-1');

        await test.step('Select annotation on media 1', async () => {
            await page.getByRole('button', { name: 'selection tool' }).click();
            await page.getByLabel('annotation rect').nth(1).click();

            const selectedAnnotations = annotatorPage.getAnnotationsList().getByLabel('selected annotation');
            await expect(selectedAnnotations).toHaveCount(1);
        });

        await test.step('Switch to media 2 and back to media 1 resets selection', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);

            await annotatorPage.selectMediaItem('item-1.jpg');
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

            const selectedAnnotations = annotatorPage.getAnnotationsList().getByLabel('selected annotation');
            await expect(selectedAnnotations).toHaveCount(0);
        });
    });

    test('Selected label persists when switching media items', async ({ page, network, annotatorPage }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-1');

        await test.step('Select non-default label on first media item', async () => {
            const blueLabelButton = page.getByRole('button', { name: `Label ${blueLabel.name}` });
            await blueLabelButton.click();

            await expect(blueLabelButton).toHaveAttribute('aria-pressed', 'true');
        });

        await test.step('Switching media keeps selected label active', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );
        });
    });

    test.describe('Annotation and prediction modes', () => {
        const predictions = [
            {
                shape: {
                    type: 'rectangle',
                    x: 3,
                    y: 0,
                    width: 780,
                    height: 421,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.9619140625],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 1007,
                    y: 624,
                    width: 909,
                    height: 456,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.9599609375],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 291,
                    y: 0,
                    width: 1553,
                    height: 889,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.904296875],
            },
        ] satisfies PredictionDTO[];

        test('Annotation vs Prediction', async ({ page, annotatorPage, boundingBoxTool, network }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([getMockedModel()]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json({
                        annotations: [],
                        user_reviewed: true,
                        subset: 'training',
                    });
                }),
                http.post('/api/projects/{project_id}/dataset/media/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: {
                                    id: '123',
                                },
                                prediction: predictions,
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('Draws an annotation in annotation mode', async () => {
                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'false');

                const annotation = { x: 100, y: 100, width: 150, height: 150 };

                await boundingBoxTool.selectTool();
                await boundingBoxTool.drawBoundingBox(annotation);

                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);

                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
            });

            await test.step('Displays server prediction in prediction mode', async () => {
                await annotatorPage.openPredictionMode();

                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'false');

                await expect(annotatorPage.getPrimaryToolbar()).toBeHidden();
                await expect(page.getByLabel('Labels', { exact: true })).toBeHidden();

                await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(predictions.length);
                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(predictions.length);

                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(0);
            });

            await test.step('Hides predictions in annotation mode, restores annotation', async () => {
                await annotatorPage.openAnnotationMode();

                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
            });

            await test.step('Edits prediction by overwriting existing annotations with prediction in annotation mode', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.editPrediction();

                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'false');

                await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(predictions.length);
                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeHidden();
            });
        });

        test('Automatically switches to annotation mode when there are annotations, no matter predictions', async ({
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([getMockedModel()]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json({
                        annotations: [
                            {
                                shape: {
                                    type: 'rectangle',
                                    x: 1007,
                                    y: 624,
                                    width: 909,
                                    height: 456,
                                },
                                labels: [{ id: redLabel.id }],
                            },
                        ],
                        user_reviewed: true,
                        subset: 'training',
                    });
                }),
                http.post('/api/projects/{project_id}/dataset/media/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: {
                                    id: '123',
                                },
                                prediction: predictions,
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');
            await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'false');
        });

        test('Displays "No object" when media:predict returns empty predictions', async ({
            page,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([getMockedModel()]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json(
                        {
                            // @ts-expect-error We care only about mocking detail
                            detail: 'Media has not been annotated yet',
                        },
                        { status: 404 }
                    );
                }),
                http.post('/api/projects/{project_id}/dataset/media/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: { id: '123' },
                                prediction: [],
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
            await expect(page.getByLabel('label No object background')).toHaveCount(1);
        });

        test('Automatically switches to prediction mode only when there are no annotations and there are predictions', async ({
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([getMockedModel()]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json(
                        {
                            // @ts-expect-error We care only about mocking detail
                            detail: 'Media has not been annotated yet',
                        },
                        { status: 404 }
                    );
                }),
                http.post('/api/projects/{project_id}/dataset/media/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: {
                                    id: '123',
                                },
                                prediction: predictions,
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'false');
            await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
        });
    });

    test('Assigns subset to media', async ({ annotatorPage, boundingBoxTool, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];
        let subsetPayload: DatasetSubset | null = 'unassigned';

        const annotationsResponsePerMedia: Record<
            string,
            { annotations: AnnotationDTO[]; user_reviewed: boolean; subset: DatasetSubset }
        > = {
            [mediaItems[0].id]: {
                annotations: [],
                user_reviewed: false,
                subset: 'unassigned',
            },
            [mediaItems[1].id]: {
                annotations: [],
                user_reviewed: false,
                subset: 'validation',
            },
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ params }) => {
                return HttpResponse.json(annotationsResponsePerMedia[params.media_id]);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.post(
                '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                async ({ request, params }) => {
                    const payload = await request.json();
                    subsetPayload = payload.subset ?? null;
                    annotationsResponsePerMedia[params.media_id].subset = payload.subset ?? 'unassigned';
                    annotationsResponsePerMedia[params.media_id].annotations = payload.annotations;

                    return HttpResponse.json({});
                }
            )
        );

        await annotatorPage.goto(mockedDetectionProject.id, mediaItems[0].id);

        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });
        });

        await test.step('Select subset', async () => {
            await annotatorPage.selectSubset('training');
        });

        await test.step('Submit annotations and subset', async () => {
            await annotatorPage.submit();

            expect(subsetPayload).toBe('training');
        });

        await test.step('Navigate to the next media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem(mediaItems[1].name);

            await expect(annotatorPage.getSelectedSubset()).toHaveText('Validation');
        });

        await test.step('Navigate to the previous media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem(mediaItems[0].name);

            await expect(annotatorPage.getSelectedSubset()).toHaveText('Training');
        });
    });
});
