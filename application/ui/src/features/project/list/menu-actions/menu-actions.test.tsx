// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { MenuActions } from './menu-actions.component';

describe('MenuActions', () => {
    const projectId = 'test-project-id';
    const projectName = 'Test Project';
    const actionButtonStyle = {};

    it('opens edit dialog when rename menu item is clicked', async () => {
        render(<MenuActions projectId={projectId} projectName={projectName} actionButtonStyle={actionButtonStyle} />);

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Rename'));

        const inputField = await screen.findByLabelText(/edit project name field/i);
        expect(inputField).toBeVisible();
        expect(inputField).toHaveValue(projectName);
    });

    it('successfully deletes project and shows success toast', async () => {
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<MenuActions projectId={projectId} projectName={projectName} actionButtonStyle={actionButtonStyle} />);

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Delete'));

        expect(await screen.findByText('Project deleted successfully')).toBeVisible();
    });

    it('shows error toast when delete project fails', async () => {
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: 'Cannot delete project' }, { status: 500 });
            })
        );

        render(<MenuActions projectId={projectId} projectName={projectName} actionButtonStyle={actionButtonStyle} />);

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Delete'));

        expect(await screen.findByText('Failed to delete project')).toBeVisible();
    });
});
