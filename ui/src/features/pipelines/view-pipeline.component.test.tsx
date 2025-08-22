// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { TestProviders } from '../../providers';
import { ViewPipeline } from './view-pipeline.component';

describe('View pipeline', () => {
    it('renders the correct values for each resource', async () => {
        server.use(
            http.get('/api/sources', () => {
                return HttpResponse.json([
                    {
                        name: 'source',
                        source_type: 'video_file',
                        video_path: 'video.mp4',
                    },
                ]);
            }),
            http.get('/api/models', () => {
                return HttpResponse.json([{ id: '1', name: 'Test-model', format: 'onnx' }]);
            }),
            http.get('/api/sinks', () => {
                return HttpResponse.json([
                    {
                        name: 'sink',
                        folder_path: 'data/sink',
                        output_formats: ['image_original', 'image_with_predictions', 'predictions'],
                        rate_limit: 0.2,
                        sink_type: 'folder',
                    },
                ]);
            })
        );

        render(
            <TestProviders>
                <ViewPipeline />
            </TestProviders>
        );

        // Headers
        expect(screen.getByRole('heading', { name: 'Source' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Sink' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();

        // Content
        expect(await screen.findByText('video_file')).toBeInTheDocument();

        expect(await screen.findByText('Test-model')).toBeInTheDocument();

        expect(await screen.findByText('data/sink')).toBeInTheDocument();
        expect(await screen.findByText('image_original,image_with_predictions,predictions')).toBeInTheDocument();
        expect(await screen.findByText('0.2')).toBeInTheDocument();
        expect(await screen.findByText('folder')).toBeInTheDocument();
    });
});
