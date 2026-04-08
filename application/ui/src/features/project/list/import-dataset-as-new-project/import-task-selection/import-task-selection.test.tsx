// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { AnnotationType, TaskType } from '../../../../../constants/shared-types';
import { server } from '../../../../../msw-node-setup';
import { ImportTaskSelection } from './import-task-selection.component';

const mockedStagedDatasetId = 'staged-dataset-123';
const mockedProject = getMockedProject({ id: 'id-1', name: 'Project 1' });

const getImportEntrySpy = vi.fn();
const updateImportEntrySpy = vi.fn();
const setCurrentStepSpy = vi.fn();

vi.mock('hooks/storage/use-import-dataset-as-new-project.hook', () => ({
    useImportDatasetAsNewProject: () => ({
        getAllImportEntries: vi.fn(),
        appendImportEntry: vi.fn(),
        deleteImportEntry: vi.fn(),
        updateImportEntryStep: vi.fn(),
        updateImportEntry: updateImportEntrySpy,
        getImportEntry: getImportEntrySpy,
    }),
}));

vi.mock('../../../providers/import-dataset-dialog-provider.component', () => ({
    useImportDatasetDialog: () => ({
        setCurrentStep: setCurrentStepSpy,
    }),
}));

describe('ImportTaskSelection', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderApp = (annotationType: AnnotationType, taskType: TaskType = 'classification') => {
        getImportEntrySpy.mockReturnValue({
            project: { name: 'Project #2', task_type: taskType },
            step: 'taskTypeSelection',
        });

        server.use(
            http.get('/api/projects', () => HttpResponse.json([mockedProject])),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: mockedStagedDatasetId,
                    format: 'geti',
                    size: 123,
                    metadata: {
                        labels: [],
                        num_images: 10,
                        num_frames: 8,
                        num_annotations: 0,
                        num_videos: 1,
                        num_annotated_frames: 5,
                        num_annotated_images: 9,
                        annotation_type: annotationType,
                    },
                    compressed: true,
                    ready_for_export: false,
                    ready_for_import: true,
                });
            })
        );

        render(<ImportTaskSelection stagedDatasetId={mockedStagedDatasetId} />);
    };

    it('shows Detection as recommended task for bounding box annotations', async () => {
        renderApp('bounding_box');

        await waitFor(() => {
            const projectNameInput = screen.getByLabelText('Project name') as HTMLInputElement;
            expect(projectNameInput.value).toBe('Project #2');
        });

        expect(await screen.findByText('Detection (Recommended)')).toBeVisible();
    });

    it('shows Instance segmentation as recommended task for polygon annotations', async () => {
        renderApp('polygon');

        await waitFor(() => {
            const projectNameInput = screen.getByLabelText('Project name') as HTMLInputElement;
            expect(projectNameInput.value).toBe('Project #2');
        });

        expect(await screen.findByText('Instance segmentation (Recommended)')).toBeVisible();
    });

    it('shows Classification as recommended task for label annotations', async () => {
        renderApp('label', 'instance_segmentation');

        await waitFor(() => {
            const projectNameInput = screen.getByLabelText('Project name') as HTMLInputElement;
            expect(projectNameInput.value).toBe('Project #2');
        });

        expect(await screen.findByText('Classification (Recommended)')).toBeVisible();
    });

    it('shows an error when the project name already exists', async () => {
        renderApp('bounding_box');

        const projectNameInput = await screen.findByLabelText('Project name');

        await userEvent.clear(projectNameInput);
        await userEvent.type(projectNameInput, `${mockedProject.name}   `);

        expect(await screen.findByText('That project name already exists')).toBeVisible();
    });
});
