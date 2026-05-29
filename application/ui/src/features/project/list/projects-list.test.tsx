// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ImportDatasetDialogProvider } from '../providers/import-dataset-dialog-provider.component';
import { ProjectList } from './project-list.component';

const renderProjectList = () => {
    return render(
        <ImportDatasetDialogProvider>
            <ProjectList />
        </ImportDatasetDialogProvider>,
        { route: '/projects', path: '/projects' }
    );
};

const projects = [
    getMockedProject({ id: 'project-1', name: 'Alpha Project', created_at: '2026-01-01T10:00:00Z' }),
    getMockedProject({ id: 'project-2', name: 'Beta Project', created_at: '2026-06-01T10:00:00Z' }),
    getMockedProject({ id: 'project-3', name: 'Zeta Project', created_at: '2026-03-01T10:00:00Z' }),
];

describe('ProjectList', () => {
    describe('with projects', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json(projects);
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('renders the "Projects" heading', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Projects' })).toBeInTheDocument();
        });

        it('renders the description text', async () => {
            renderProjectList();

            expect(
                await screen.findByText(/Create projects to configure new computer vision pipelines/i)
            ).toBeInTheDocument();
        });

        it('renders a card for each project', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();
            expect(await screen.findByRole('heading', { name: 'Beta Project' })).toBeInTheDocument();
            expect(await screen.findByRole('heading', { name: 'Zeta Project' })).toBeInTheDocument();
        });

        it('defaults to sorting by created date (newest first)', async () => {
            renderProjectList();

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Beta Project');
            expect(headings[1]).toHaveTextContent('Zeta Project');
            expect(headings[2]).toHaveTextContent('Alpha Project');
        });

        it('sorts projects by name ascending when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Name (A-Z)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Alpha Project');
            expect(headings[1]).toHaveTextContent('Beta Project');
            expect(headings[2]).toHaveTextContent('Zeta Project');
        });

        it('sorts projects by name descending when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Name (Z-A)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Zeta Project');
            expect(headings[1]).toHaveTextContent('Beta Project');
            expect(headings[2]).toHaveTextContent('Alpha Project');
        });

        it('sorts projects by created date oldest first when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Created date (oldest)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Alpha Project');
            expect(headings[1]).toHaveTextContent('Zeta Project');
            expect(headings[2]).toHaveTextContent('Beta Project');
        });

        it('each project card links to the project dataset page', async () => {
            renderProjectList();

            const alphaCard = (await screen.findByRole('heading', { name: 'Alpha Project' })).closest('a');
            expect(alphaCard).toHaveAttribute('href', '/projects/project-1/dataset');

            const betaCard = (await screen.findByRole('heading', { name: 'Beta Project' })).closest('a');
            expect(betaCard).toHaveAttribute('href', '/projects/project-2/dataset');
        });
    });

    describe('with no projects', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json([]);
                })
            );
        });

        it('shows empty illustration when there are no projects', async () => {
            renderProjectList();

            expect(await screen.findByLabelText('empty list')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /create new project/i })).toBeVisible();
            expect(screen.getByRole('button', { name: /create from dataset/i })).toBeVisible();

            expect(screen.queryByRole('button', { name: /sort/i })).not.toBeInTheDocument();
        });
    });

    describe('create project card', () => {
        it('renders create new project and create project from dataset buttons', async () => {
            server.use(http.get('/api/projects', () => HttpResponse.json(projects)));

            renderProjectList();

            const createButton = await screen.findByRole('button', { name: /create new project/i });
            expect(createButton).toBeVisible();

            const createFromDatasetButton = await screen.findByRole('button', { name: /Create project from dataset/i });
            expect(createFromDatasetButton).toBeVisible();
        });
    });
});
