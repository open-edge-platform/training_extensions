// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { paths } from '../../../constants/paths';
import { server } from '../../../msw-node-setup';
import { ModelListingContainer } from './model-listing.container';

const PROJECT_ID = '7b073838-99d3-42ff-9018-4e901eb047fc';

const renderModelListing = () => {
    return render(<ModelListingContainer />, {
        route: paths.project.models({ projectId: PROJECT_ID }),
        path: paths.project.models.pattern,
    });
};

describe('ModelListingContainer', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({ id: PROJECT_ID }))),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/dataset_revisions', () =>
                HttpResponse.json([getMockedDatasetRevision({ id: 'dataset-revision-1', name: 'Dataset Revision 1' })])
            )
        );
    });

    it('hiding failed models when only failed models exist does not show the "No models yet" screen', async () => {
        const user = userEvent.setup();

        server.use(
            http.get('/api/projects/{project_id}/models', () =>
                HttpResponse.json([
                    getMockedModel({ id: 'model-1', name: 'Failed YOLOX Model', training_info: { status: 'failed' } }),
                ])
            )
        );

        renderModelListing();

        expect(await screen.findByText('Failed YOLOX Model')).toBeInTheDocument();
        expect(screen.queryByText(/No models yet\./i)).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Model listing options' }));

        await user.click(screen.getByRole('menuitem', { name: /hide failed models/i }));

        expect(screen.queryByText(/No models yet\./i)).not.toBeInTheDocument();

        expect(screen.getByText('No models found')).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'Model listing options' })).toBeInTheDocument();
    });

    it('shows the "No models yet" screen when the project has no models at all', async () => {
        server.use(http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])));

        renderModelListing();

        expect(await screen.findByText(/No models yet\./i)).toBeInTheDocument();
        expect(screen.getByText(/Train your first model to get started\./i)).toBeInTheDocument();
    });
});
