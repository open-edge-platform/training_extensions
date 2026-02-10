// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { DeleteProjectDialog } from './delete-project-dialog.component';

describe('DeleteProjectDialog', () => {
    const projectId = 'test-project-id';
    const projectName = 'Test Project';

    it('successfully deletes project and shows success toast', async () => {
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(
            <DeleteProjectDialog projectId={projectId} projectName={projectName} isOpen={true} onClose={() => {}} />
        );

        fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('Project deleted successfully')).toBeVisible();
    });

    it('shows error toast when delete project fails', async () => {
        const errorMessage = 'Cannot delete project';
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(
            <DeleteProjectDialog projectId={projectId} projectName={projectName} isOpen={true} onClose={() => {}} />
        );

        fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

        expect(await screen.findByText(errorMessage)).toBeVisible();
    });
});
