// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { Project } from '../../../constants/shared-types';
import { server } from '../../../msw-node-setup';
import { CreateProjectForm } from './create-project-form';

const mockNavigate = vi.hoisted(() => vi.fn());

vi.mock('react-router-dom', async (importOriginal) => ({
    ...(await importOriginal()),
    useNavigate: () => mockNavigate,
}));

const renderCreateProjectForm = (projects: Project[] = []) => render(<CreateProjectForm projects={projects} />);

const selectTask = (taskLabel: string) => {
    fireEvent.click(screen.getByLabelText(`Task option: ${taskLabel}`));
};

const selectClassificationType = (type: 'Single-label' | 'Multi-label') => {
    fireEvent.click(screen.getByRole('radio', { name: type }));
};

const addLabel = async (name: string) => {
    await userEvent.type(screen.getByRole('textbox', { name: 'Create label input' }), name);
    fireEvent.click(screen.getByRole('button', { name: /create label/i }));
};

const getCreateButton = () => screen.getByRole('button', { name: /create project/i });

const clickGoBack = () => {
    fireEvent.click(screen.getByRole('button', { name: /go back/i }));
};

describe('CreateProjectForm', () => {
    beforeEach(() => {
        vi.resetAllMocks();
    });

    describe('initial state', () => {
        it('disables create button when no task type is selected', () => {
            renderCreateProjectForm();

            expect(getCreateButton()).toBeDisabled();
        });
    });

    describe('single-label classification', () => {
        it('disables create button with zero labels', () => {
            renderCreateProjectForm();

            selectTask('Image Classification');

            expect(getCreateButton()).toBeDisabled();
        });

        it('disables create button with exactly one label', async () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            await addLabel('Cat');

            expect(getCreateButton()).toBeDisabled();
        });

        it('enables create button with two labels', async () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            await addLabel('Cat');
            await addLabel('Dog');

            expect(getCreateButton()).toBeEnabled();
        });

        it('enables create button with more than two labels', async () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            await addLabel('Cat');
            await addLabel('Dog');
            await addLabel('Bird');

            expect(getCreateButton()).toBeEnabled();
        });
    });

    describe('multi-label classification', () => {
        it('disables create button with zero labels', () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            selectClassificationType('Multi-label');

            expect(getCreateButton()).toBeDisabled();
        });

        it('enables create button with one label', async () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            selectClassificationType('Multi-label');
            await addLabel('Cat');

            expect(getCreateButton()).toBeEnabled();
        });

        it('enables create button with two or more labels', async () => {
            renderCreateProjectForm();

            selectTask('Image Classification');
            selectClassificationType('Multi-label');
            await addLabel('Cat');
            await addLabel('Dog');

            expect(getCreateButton()).toBeEnabled();
        });
    });

    describe('detection and segmentation tasks', () => {
        it.each([
            { taskLabel: 'Object Detection', labelName: 'Car' },
            { taskLabel: 'Instance Segmentation', labelName: 'Tree' },
        ])('disables create button with zero labels for $taskLabel', ({ taskLabel }) => {
            renderCreateProjectForm();

            selectTask(taskLabel);

            expect(getCreateButton()).toBeDisabled();
        });

        it.each([
            { taskLabel: 'Object Detection', labelName: 'Car' },
            { taskLabel: 'Instance Segmentation', labelName: 'Tree' },
        ])('enables create button with one label for $taskLabel', async ({ taskLabel, labelName }) => {
            renderCreateProjectForm();

            selectTask(taskLabel);
            await addLabel(labelName);

            expect(getCreateButton()).toBeEnabled();
        });
    });

    describe('project name validation', () => {
        it('disables create button when project name is empty', async () => {
            renderCreateProjectForm();

            selectTask('Object Detection');
            await addLabel('Car');

            const nameInput = screen.getByRole('textbox', { name: 'Project name input' });
            await userEvent.clear(nameInput);

            expect(getCreateButton()).toBeDisabled();
        });

        it('disables create button when project name is a duplicate', async () => {
            const existingProject = getMockedProject({ name: 'My Existing Project' });
            renderCreateProjectForm([existingProject]);

            selectTask('Object Detection');
            await addLabel('Car');

            const nameInput = screen.getByRole('textbox', { name: 'Project name input' });
            await userEvent.clear(nameInput);
            await userEvent.type(nameInput, 'My Existing Project');

            expect(getCreateButton()).toBeDisabled();
        });
    });

    describe('form submission', () => {
        it('submits single-label classification with exclusive_labels set to true', async () => {
            let capturedBody: Record<string, unknown> = {};

            server.use(
                http.post('/api/projects', async ({ request }) => {
                    capturedBody = (await request.json()) as Record<string, unknown>;
                    return HttpResponse.json(getMockedProject());
                })
            );

            renderCreateProjectForm();

            selectTask('Image Classification');
            await addLabel('Cat');
            await addLabel('Dog');

            fireEvent.click(getCreateButton());

            await waitFor(() => {
                expect(capturedBody).toMatchObject({
                    task: { task_type: 'classification', exclusive_labels: true },
                });
            });
        });

        it('submits multi-label classification with exclusive_labels set to false', async () => {
            let capturedBody: Record<string, unknown> = {};

            server.use(
                http.post('/api/projects', async ({ request }) => {
                    capturedBody = (await request.json()) as Record<string, unknown>;
                    return HttpResponse.json(getMockedProject());
                })
            );

            renderCreateProjectForm();

            selectTask('Image Classification');
            selectClassificationType('Multi-label');
            await addLabel('Cat');

            fireEvent.click(getCreateButton());

            await waitFor(() => {
                expect(capturedBody).toMatchObject({
                    task: { task_type: 'classification', exclusive_labels: false },
                });
            });
        });

        it.each([
            { taskLabel: 'Object Detection', taskType: 'detection', labelName: 'Car' },
            { taskLabel: 'Instance Segmentation', taskType: 'instance_segmentation', labelName: 'Tree' },
        ])('submits $taskLabel project with correct task_type', async ({ taskLabel, taskType, labelName }) => {
            let capturedBody: Record<string, unknown> = {};

            server.use(
                http.post('/api/projects', async ({ request }) => {
                    capturedBody = (await request.json()) as Record<string, unknown>;
                    return HttpResponse.json(getMockedProject());
                })
            );

            renderCreateProjectForm();

            selectTask(taskLabel);
            await addLabel(labelName);

            fireEvent.click(getCreateButton());

            await waitFor(() => {
                expect(capturedBody).toMatchObject({
                    task: { task_type: taskType, exclusive_labels: false },
                });
            });
        });
    });

    describe('navigation', () => {
        test('navigates back when "Go Back" button is clicked', () => {
            renderCreateProjectForm();

            clickGoBack();

            expect(mockNavigate).toHaveBeenCalledWith(-1);
        });
    });
});
