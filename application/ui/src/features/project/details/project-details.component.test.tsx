// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';
import { render, screen } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ProjectDetails } from './project-details.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

describe('ProjectDetails', () => {
    it('renders the correct values for each resource', async () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    id: '123',
                    name: 'Test Project',
                    task: {
                        task_type: 'detection',
                        exclusive_labels: false,
                        labels: [
                            { id: '1', color: 'red', name: 'person' },
                            { id: '2', color: 'blue', name: 'car' },
                        ],
                    },
                    active_pipeline: true,
                });
            }),
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json({
                    project_id: '123',
                    status: 'running' as const,
                    data_collection_policies: [],
                    source: {
                        id: 'source-id',
                        name: 'source',
                        source_type: 'video_file' as const,
                        video_path: 'video.mp4',
                    },
                    model: {
                        id: '1',
                        name: 'My amazing model',
                        architecture: 'Object_Detection_TestModel',
                        training_info: {
                            status: 'successful' as const,
                            label_schema_revision: {},
                            configuration: {},
                        },
                        files_deleted: false,
                    },
                    sink: {
                        id: 'sink-id',
                        name: 'sink',
                        folder_path: 'data/sink',
                        output_formats: ['image_original', 'image_with_predictions', 'predictions'] as Array<
                            'image_original' | 'image_with_predictions' | 'predictions'
                        >,
                        rate_limit: 0.2,
                        sink_type: 'folder' as const,
                    },
                    device: 'cpu',
                } satisfies Record<string, unknown>);
            })
        );

        render(<ProjectDetails />);

        // Wait for the component to load and render the Project heading
        expect(await screen.findByRole('heading', { name: 'Project' })).toBeInTheDocument();
        expect(screen.getAllByRole('heading', { name: 'Name' })[0]).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Task type' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Labels' })).toBeInTheDocument();

        // Project content
        expect(screen.getByText('Test Project')).toBeInTheDocument();
        expect(screen.getByText('detection')).toBeInTheDocument();
        expect(screen.getByText('person, car')).toBeInTheDocument();

        // Pipeline section headers
        expect(await screen.findByRole('heading', { name: 'Pipeline' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Source' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Sink' })).toBeInTheDocument();

        // Pipeline content
        expect(screen.getByText('video_file')).toBeInTheDocument();
        expect(screen.getByText('video.mp4')).toBeInTheDocument();

        expect(screen.getByText('Object_Detection_TestModel')).toBeInTheDocument();

        expect(screen.getByText('data/sink')).toBeInTheDocument();
        expect(screen.getByText('image_original,image_with_predictions,predictions')).toBeInTheDocument();
        expect(screen.getByText('0.2')).toBeInTheDocument();
        expect(screen.getByText('folder')).toBeInTheDocument();
    });
});
