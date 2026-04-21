// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { ExportDatasetMetadata } from '../../../../../../constants/shared-types';
import { server } from '../../../../../../msw-node-setup';
import { ExportJobDetails } from './export-details.component';

describe('ExportJobDetails', () => {
    const renderApp = (metadata: ExportDatasetMetadata) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(
                    getMockedProject({
                        id: 'project-123',
                        task: {
                            labels: [
                                getMockedLabel({ id: 'label-1', name: 'label 1' }),
                                getMockedLabel({ id: 'label-2', name: 'label 2' }),
                                getMockedLabel({ id: 'label-3', name: 'label 3' }),
                            ],
                            task_type: 'detection',
                            exclusive_labels: false,
                        },
                    })
                );
            })
        );

        render(<ExportJobDetails metadata={metadata} />);
    };

    it('displays all labels when no specific labels are filtered', async () => {
        const metadata: ExportDatasetMetadata = {
            dataset_id: 'dataset-123',
            project_id: 'project-123',
            export_format: 'COCO',
            filters: {
                include_unannotated: true,
            },
        };

        renderApp(metadata);

        expect(await screen.findByText(/Labels:\s*label 1, label 2, label 3/)).toBeVisible();
    });

    it('displays selected labels when specific labels are filtered', async () => {
        const metadata: ExportDatasetMetadata = {
            dataset_id: 'dataset-123',
            project_id: 'project-123',
            export_format: 'YOLO',
            filters: {
                labels: ['label 1', 'label 2'],
                include_unannotated: true,
            },
        };

        renderApp(metadata);

        expect(await screen.findByText(/Labels:\s*label 1, label 2$/)).toBeVisible();
        expect(screen.queryByText(/label 3/)).not.toBeInTheDocument();
    });

    it('filters out labels that are not in the project', async () => {
        const metadata: ExportDatasetMetadata = {
            dataset_id: 'dataset-123',
            project_id: 'project-123',
            export_format: 'COCO',
            filters: {
                labels: ['label 1', 'NonExistentLabel'],
                include_unannotated: true,
            },
        };

        renderApp(metadata);

        expect(await screen.findByText(/Labels:\s*label 1$/)).toBeVisible();
        expect(screen.queryByText(/NonExistentLabel/)).not.toBeInTheDocument();
    });

    it('displays "Only media with annotations" when include_unannotated is false', async () => {
        const metadata: ExportDatasetMetadata = {
            dataset_id: 'dataset-123',
            project_id: 'project-123',
            export_format: 'COCO',
            filters: { include_unannotated: false },
        };

        renderApp(metadata);

        expect(await screen.findByText(/Only media with annotations/)).toBeVisible();
    });

    it('does not display "Only media with annotations" when include_unannotated is true', async () => {
        const metadata: ExportDatasetMetadata = {
            dataset_id: 'dataset-123',
            project_id: 'project-123',
            export_format: 'COCO',
            filters: { include_unannotated: true },
        };

        renderApp(metadata);

        await screen.findByText('COCO');

        expect(await screen.findByText(/All media/)).toBeVisible();
    });
});
