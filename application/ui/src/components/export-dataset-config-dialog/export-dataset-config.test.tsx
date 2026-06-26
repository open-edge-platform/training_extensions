// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedProject } from '../../../mocks/mock-project';
import { SchemaProjectView } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { ExportDatasetConfig } from './export-dataset-config.component';

describe('ExportDatasetConfig', () => {
    const mockDialogState = {
        isOpen: true,
        open: vi.fn(),
        close: vi.fn(),
        toggle: vi.fn(),
        setOpen: vi.fn(),
    };

    const VIDEO_WARNING = /Exporting videos is not supported by this dataset format/i;
    const EMPTY_LABEL_WARNING_NO_OBJECT = /Empty labels \('No object'\) are exclusively supported by the Geti format/i;
    const EMPTY_LABEL_WARNING_NO_LABEL = /Empty labels \('No label'\) are exclusively supported by the Geti format/i;

    const renderApp = (project: SchemaProjectView) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(project);
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: { total: 0, offset: 0, limit: 0, count: 0 },
                    items: [],
                });
            })
        );

        render(<ExportDatasetConfig dialogState={mockDialogState} datasetId={null} statistics={undefined} />);
    };

    it('shows only GETI export option for classification task', async () => {
        renderApp(
            getMockedProject({
                task: { exclusive_labels: true, task_type: 'classification' },
            })
        );

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeVisible();
        expect(screen.queryByRole('radio', { name: 'YOLO' })).not.toBeInTheDocument();
        expect(screen.queryByRole('radio', { name: 'COCO' })).not.toBeInTheDocument();
    });

    it('shows GETI and COCO export option for instance_segmentation task', async () => {
        renderApp(
            getMockedProject({
                task: { exclusive_labels: true, task_type: 'instance_segmentation' },
            })
        );

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeVisible();
        expect(screen.queryByRole('radio', { name: 'COCO' })).toBeVisible();
    });

    it('does not show the video export warning when the default Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'instance_segmentation' } }));

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(VIDEO_WARNING)).not.toBeInTheDocument();
    });

    it('shows the video export warning when a non-Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'instance_segmentation' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(screen.getByRole('radio', { name: 'COCO' })).toBeChecked();
        expect(screen.getByText(VIDEO_WARNING)).toBeVisible();

        fireEvent.click(screen.getByRole('radio', { name: 'Geti' }));
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(VIDEO_WARNING)).not.toBeInTheDocument();
    });

    it('shows the empty label warning for a task with empty labels when a non-Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(screen.getByRole('radio', { name: 'COCO' })).toBeChecked();
        expect(screen.getByText(EMPTY_LABEL_WARNING_NO_OBJECT)).toBeVisible();

        fireEvent.click(screen.getByRole('radio', { name: 'Geti' }));
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_OBJECT)).not.toBeInTheDocument();
    });

    it('does not show the empty label warning for a single-label classification task', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'classification' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'VOC' }));
        expect(screen.getByRole('radio', { name: 'VOC' })).toBeChecked();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_OBJECT)).not.toBeInTheDocument();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_LABEL)).not.toBeInTheDocument();
    });
});
