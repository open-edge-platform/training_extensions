import { render, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { TestProviders } from '../../providers';
import { Index as PipelineIndex } from './index';

describe('View pipeline', () => {
    it('renders the pipeline view with input, model, and output sections', () => {
        render(
            <TestProviders>
                <PipelineIndex />
            </TestProviders>
        );

        expect(screen.getByRole('heading', { name: 'Input' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Output' })).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
    });

    it('renders the correct values for each resource', () => {
        server.use(
            http.get('/api/inputs', (_info) =>
                HttpResponse.json({
                    name: 'source',
                    source_type: 'video_file',
                    video_path: 'video.mp4',
                })
            ),
            http.get('/api/models', (_info) =>
                HttpResponse.json({ active_model: 'test-model', available_models: ['test-model'] })
            ),
            http.get('/api/outputs', (_info) =>
                HttpResponse.json({
                    name: 'output',
                    folder_path: 'data/output',
                    output_formats: ['image_original', 'image_with_predictions', 'predictions'],
                    rate_limit: 0.2,
                    sink_type: 'folder',
                })
            )
        );

        render(
            <TestProviders>
                <PipelineIndex />
            </TestProviders>
        );

        expect(screen.getByText('Default name')).toBeInTheDocument();
        expect(screen.getByText('video_file')).toBeInTheDocument();

        expect(screen.getByText('test-model')).toBeInTheDocument();

        expect(screen.getByText('data/output')).toBeInTheDocument();
        expect(screen.getByText('image_original')).toBeInTheDocument();
        expect(screen.getByText('image_with_predictions')).toBeInTheDocument();
        expect(screen.getByText('predictions')).toBeInTheDocument();
        expect(screen.getByText('0.2')).toBeInTheDocument();
        expect(screen.getByText('folder')).toBeInTheDocument();
    });
});
