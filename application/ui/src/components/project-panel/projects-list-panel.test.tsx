// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { ProjectsListPanel } from './projects-list-panel.component';

const selectedProjectId = 'selected-project-id';
const selectedProjectName = 'Selected Project';
const otherProjectId = 'other-project-id';
const otherProjectName = 'Other Project';

const selectedProject = getMockedProject({ id: selectedProjectId, name: selectedProjectName, active_pipeline: false });
const otherProject = getMockedProject({ id: otherProjectId, name: otherProjectName, active_pipeline: false });

const mockNavigate = vi.hoisted(() => vi.fn());

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return { ...actual, useNavigate: () => mockNavigate };
});

describe('ProjectsListPanel', () => {
    beforeEach(() => {
        mockNavigate.mockClear();
        server.use(
            http.get('/api/projects', () => HttpResponse.json([selectedProject, otherProject])),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(getMockedPipeline()))
        );
    });

    const renderPanel = () => {
        return render(<ProjectsListPanel />, {
            route: `/projects/${selectedProjectId}`,
        });
    };

    const openSelector = async (user: ReturnType<typeof userEvent.setup>) => {
        const selectorButton = await screen.findByRole('button', {
            name: new RegExp(`Selected project ${selectedProjectName}`, 'i'),
        });
        await user.click(selectorButton);
        await screen.findByText('Manage projects');
    };

    it('rename from another project row closes selector and opens rename dialog prefilled with project name', async () => {
        const user = userEvent.setup();
        renderPanel();
        await openSelector(user);

        const otherProjectMenuButton = screen.getByTestId(otherProjectId);
        await user.click(otherProjectMenuButton);

        const renameOption = await screen.findByText('Rename');
        await user.click(renameOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        const inputField = await screen.findByLabelText(/edit project name field/i);
        expect(inputField).toBeVisible();
        expect(inputField).toHaveValue(otherProjectName);
    });

    it('blocked enable pipeline from selector closes selector and shows explanation dialog', async () => {
        server.use(http.get('/api/projects/{project_id}/pipeline', () => new HttpResponse(null, { status: 404 })));

        const user = userEvent.setup();
        renderPanel();
        await openSelector(user);

        const otherProjectMenuButton = screen.getByTestId(otherProjectId);
        await user.click(otherProjectMenuButton);

        const enableOption = await screen.findByText('Enable pipeline');
        await user.click(enableOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model and source before enabling the pipeline.')
        ).toBeVisible();
    });

    it('delete from project row closes selector, shows confirmation, and navigates when deleting selected project', async () => {
        const deletedProjectIds: string[] = [];
        server.use(
            http.delete('/api/projects/{project_id}', ({ params }) => {
                deletedProjectIds.push(params.project_id);
                return HttpResponse.json(null, { status: 204 });
            })
        );

        const user = userEvent.setup();
        renderPanel();
        await openSelector(user);

        const selectedProjectMenuButton = screen.getByTestId(selectedProjectId);
        await user.click(selectedProjectMenuButton);

        const deleteOption = await screen.findByText('Delete');
        await user.click(deleteOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        expect(
            await screen.findByText(`Are you sure you want to delete project "${selectedProjectName}"?`)
        ).toBeVisible();

        const confirmDeleteButton = await screen.findByRole('button', { name: 'Delete' });
        await user.click(confirmDeleteButton);

        await waitFor(() => {
            expect(deletedProjectIds).toContain(selectedProjectId);
        });

        await waitFor(() => {
            expect(mockNavigate).toHaveBeenCalledWith('/projects');
        });
    });
});
