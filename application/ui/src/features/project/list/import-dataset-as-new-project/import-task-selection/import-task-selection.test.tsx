// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { AnnotationType, TaskType } from '../../../../../constants/shared-types';
import { server } from '../../../../../msw-node-setup';
import { ImportTaskSelection } from './import-task-selection.component';

const mockedStagedDatasetId = 'staged-dataset-123';

const getImportEntrySpy = vi.fn();
const updateImportEntrySpy = vi.fn();
const setCurrentStepSpy = vi.fn();

vi.mock('hooks/localStorage/use-import-dataset-as-new-project.hook', () => ({
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
            project: {
                name: 'Project #2',
                task_type: taskType,
            },
            step: 'taskTypeSelection',
        });

        server.use(
            http.get('/api/projects', () => HttpResponse.json([getMockedProject({ id: 'id-1', name: 'Project 1' })])),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: mockedStagedDatasetId,
                    format: 'geti',
                    size: 123,
                    metadata: {
                        labels: [],
                        num_items: 0,
                        num_annotations: 0,
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
});
