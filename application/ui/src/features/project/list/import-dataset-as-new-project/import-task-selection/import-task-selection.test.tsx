// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { AnnotationType, DatasetFormat, TaskType } from '../../../../../constants/shared-types';
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

    const renderApp = ({
        format = 'geti',
        taskType,
        annotationType,
    }: {
        format: DatasetFormat;
        taskType?: TaskType;
        annotationType: AnnotationType;
    }) => {
        getImportEntrySpy.mockReturnValue({
            step: 'taskTypeSelection',
            project: { name: 'Project #2', task_type: taskType },
        });

        server.use(
            http.get('/api/projects', () => HttpResponse.json([mockedProject])),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: mockedStagedDatasetId,
                    format,
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

    it('shows only Object detection as recommended for bounding box annotations in geti format', async () => {
        renderApp({ format: 'geti', taskType: 'detection', annotationType: 'bounding_box' });

        expect(await screen.findByRole('button', { name: /Task type/i })).toHaveTextContent('Object detection (Recommended)');
    });

    it('shows only Instance segmentation as recommended for polygon annotations in geti format', async () => {
        renderApp({ format: 'geti', taskType: 'instance_segmentation', annotationType: 'polygon' });

        expect(await screen.findByRole('button', { name: /Task type/i })).toHaveTextContent(
            'Instance segmentation (Recommended)'
        );
        expect(screen.getByText('Object detection')).toBeVisible();
        expect(screen.getByText('Classification')).toBeVisible();
    });

    it('shows only Classification as recommended for label annotations in geti format', async () => {
        renderApp({ format: 'geti', taskType: 'classification', annotationType: 'label' });

        expect(await screen.findByRole('button', { name: /Task type/i })).toHaveTextContent(
            'Classification (Recommended)'
        );
    });

    it('does not show recommended task types for non-geti format', async () => {
        renderApp({ format: 'coco', annotationType: 'bounding_box' });

        expect(await screen.findByRole('button', { name: /Task type/i })).toHaveTextContent(/Select an option.../i);
    });

    it('shows an error when the project name already exists', async () => {
        renderApp({ format: 'geti', taskType: 'detection', annotationType: 'bounding_box' });

        const projectNameInput = await screen.findByLabelText('Project name');

        await userEvent.clear(projectNameInput);
        await userEvent.type(projectNameInput, `${mockedProject.name}   `);

        expect(await screen.findByText('That project name already exists')).toBeVisible();
    });
});
