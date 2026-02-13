// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedProject } from '../../../mocks/mock-project';
import { SchemaProjectView } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { ExportDataset } from './export-dataset-config.component';

describe('ExportDataset', () => {
    const mockDialogState = {
        isOpen: true,
        open: vi.fn(),
        close: vi.fn(),
        toggle: vi.fn(),
        setOpen: vi.fn(),
    };

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

        render(<ExportDataset dialogState={mockDialogState} />);
    };

    it('hides COCO option for classification task', async () => {
        renderApp(
            getMockedProject({
                task: { exclusive_labels: true, task_type: 'classification' },
            })
        );

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'GETI' })).toBeVisible();
        expect(screen.getByRole('radio', { name: 'YOLO' })).toBeVisible();
        expect(screen.queryByRole('radio', { name: 'COCO' })).not.toBeInTheDocument();
    });
});
