// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { Model, Pipeline } from '../../../constants/shared-types';
import { server } from '../../../msw-node-setup';
import { ActiveModel } from './active-model.component';

const successfulModelA = getMockedModel({ id: 'model-a', name: 'Model A' });
const successfulModelB = getMockedModel({ id: 'model-b', name: 'Model B' });
const runningModelB = getMockedModel({
    id: 'running-model',
    name: 'Running model',
    training_info: {
        ...getMockedModel().training_info,
        status: 'in_progress',
    },
});
const failedModel = getMockedModel({
    id: 'model-failed',
    name: 'Failed Model',
    training_info: {
        ...getMockedModel().training_info,
        status: 'failed',
    },
});
const notStartedModel = getMockedModel({
    id: 'not-started-model',
    name: 'Not started model',
    training_info: {
        ...getMockedModel().training_info,
        status: 'not_started',
    },
});
const modelWithDeletedFiles = getMockedModel({
    id: 'model-deleted-files',
    name: 'Model with deleted files',
    training_info: {
        ...getMockedModel().training_info,
        status: 'successful',
    },
    files_deleted: true,
});

const modelWithOpenVinoVariant = getMockedModel({
    id: 'model-with-ov',
    name: 'Model A',
    variants: [
        {
            id: 'variant-ov-fp16',
            format: 'openvino',
            precision: 'fp16',
            weights_size: 0,
            evaluations: [],
            files_deleted: false,
        },
    ],
});

const modelWithOnnxVariant = getMockedModel({
    id: 'model-with-onnx',
    name: 'Model Onnx',
    variants: [
        {
            id: 'variant-onnx',
            format: 'onnx',
            precision: 'fp32',
            weights_size: 0,
            evaluations: [],
            files_deleted: false,
        },
    ],
});

describe('ActiveModel', () => {
    const renderApp = ({ models, pipeline }: { models: Model[]; pipeline: Pipeline }) => {
        const patchSpy = vi.fn();

        server.use(
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json(models)),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(pipeline)),
            http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                patchSpy(await request.json());

                return HttpResponse.json({
                    project_id: '',
                    status: 'idle',
                    device: 'images_folder',
                });
            })
        );

        render(<ActiveModel />);

        return patchSpy;
    };

    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('shows only successful models', async () => {
        renderApp({
            models: [
                successfulModelA,
                successfulModelB,
                runningModelB,
                failedModel,
                notStartedModel,
                modelWithDeletedFiles,
            ],
            pipeline: getMockedPipeline(),
        });

        expect(await screen.findByLabelText('active model')).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));

        expect(await screen.findByRole('option', { name: 'Model A' })).toBeVisible();
        expect(screen.getByRole('option', { name: 'Model B' })).toBeVisible();
        expect(screen.queryByRole('option', { name: 'Running model' })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'Not started model' })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'Failed Model' })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'Model with deleted files' })).not.toBeInTheDocument();
    });

    it('patches pipeline upon change', async () => {
        const patchSpy = renderApp({
            models: [successfulModelA, successfulModelB],
            pipeline: getMockedPipeline(),
        });

        await screen.findByLabelText('active model');

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));
        fireEvent.click(await screen.findByRole('option', { name: 'Model B' }));

        await waitFor(() => {
            expect(patchSpy).toHaveBeenCalledWith({ model_id: 'model-b' });
        });
    });

    it('does not render model picker when there are no models', async () => {
        renderApp({ models: [], pipeline: getMockedPipeline() });

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        await waitFor(() => {
            expect(screen.queryByRole('button', { name: /active model/i })).not.toBeInTheDocument();
        });
    });

    it('shows openvino variant entries in the picker', async () => {
        renderApp({
            models: [modelWithOpenVinoVariant],
            pipeline: getMockedPipeline(),
        });

        expect(await screen.findByLabelText('active model')).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));

        expect(await screen.findByRole('option', { name: 'Model A' })).toBeVisible();
        expect(screen.getByRole('option', { name: 'Model A [FP16]' })).toBeVisible();
    });

    it('does not show non-openvino variants', async () => {
        renderApp({
            models: [modelWithOnnxVariant],
            pipeline: getMockedPipeline(),
        });

        expect(await screen.findByLabelText('active model')).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));

        expect(await screen.findByRole('option', { name: 'Model Onnx' })).toBeVisible();
        expect(screen.queryByRole('option', { name: 'Model Onnx [FP32]' })).not.toBeInTheDocument();
    });

    it('patches pipeline with variant id when openvino variant is selected', async () => {
        const patchSpy = renderApp({
            models: [modelWithOpenVinoVariant],
            pipeline: getMockedPipeline(),
        });

        await screen.findByLabelText('active model');

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));
        fireEvent.click(await screen.findByRole('option', { name: 'Model A [FP16]' }));

        await waitFor(() => {
            expect(patchSpy).toHaveBeenCalledWith({ model_id: 'variant-ov-fp16' });
        });
    });
});
