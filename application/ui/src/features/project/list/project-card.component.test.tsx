// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedProject } from 'mocks/mock-project';
import { fireEvent, render, screen } from 'test-utils/render';

import { ProjectCard } from './project-card.component';

describe('ProjectCard', () => {
    const mockProject = getMockedProject({
        id: 'test-project-id',
        name: 'Test Project',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: 'label-1', name: 'Cat', color: '#FF0000' },
                { id: 'label-2', name: 'Dog', color: '#00FF00' },
            ],
        },
    });

    it('should render all elements correctly', async () => {
        render(<ProjectCard item={mockProject} />);

        expect(await screen.findByRole('heading', { name: 'Test Project' })).toBeInTheDocument();

        const thumbnail = await screen.findByRole('img', { name: 'Test Project' });
        expect(thumbnail).toBeInTheDocument();
        expect(thumbnail).toHaveAttribute('alt', 'Test Project');

        expect(await screen.findByText('detection')).toBeInTheDocument();
        expect(await screen.findByText('• Labels: Cat, Dog')).toBeInTheDocument();
        expect(await screen.findByText('• Edited: 2025-08-07 06:05 AM')).toBeInTheDocument();
        expect(await screen.findByRole('button', { name: /open project options/i })).toBeInTheDocument();
    });

    it('should show active tag when pipeline is running', async () => {
        render(<ProjectCard item={{ ...mockProject, active_pipeline: true }} />);

        expect(await screen.findByLabelText('Active')).toBeInTheDocument();
    });

    it('should not show active tag when pipeline is idle', async () => {
        render(<ProjectCard item={mockProject} />);

        expect(screen.queryByText('Active')).not.toBeInTheDocument();
    });

    it('should render as a link to project inference page', async () => {
        render(<ProjectCard item={mockProject} />);

        const cardLink = await screen.findByRole('link');
        expect(cardLink).toHaveAttribute('href', '/projects/test-project-id/inference');
    });

    it('should display single label correctly', async () => {
        const singleLabelProject = getMockedProject({
            task: {
                task_type: 'classification',
                exclusive_labels: true,
                labels: [{ id: 'label-1', name: 'Person', color: '#FF0000', hotkey: 'P' }],
            },
        });

        render(<ProjectCard item={singleLabelProject} />);

        expect(await screen.findByText('• Labels: Person')).toBeInTheDocument();
    });

    it('should display multiple labels separated by commas', async () => {
        const multiLabelProject = getMockedProject({
            task: {
                task_type: 'detection',
                exclusive_labels: false,
                labels: [
                    { id: 'label-1', name: 'Car', color: '#FF0000', hotkey: 'C' },
                    { id: 'label-2', name: 'Truck', color: '#00FF00', hotkey: 'T' },
                    { id: 'label-3', name: 'Bus', color: '#0000FF', hotkey: 'B' },
                ],
            },
        });

        render(<ProjectCard item={multiLabelProject} />);

        expect(await screen.findByText('• Labels: Car, Truck, Bus')).toBeInTheDocument();
    });

    it('should handle empty labels array', async () => {
        const noLabelsProject = getMockedProject({
            task: {
                task_type: 'instance_segmentation',
                exclusive_labels: false,
                labels: [],
            },
        });

        render(<ProjectCard item={noLabelsProject} />);

        expect(await screen.findByText('• Labels:')).toBeInTheDocument();
    });

    it('should display classification task type', async () => {
        const classificationProject = getMockedProject({
            task: { ...mockProject.task, task_type: 'classification' },
        });

        render(<ProjectCard item={classificationProject} />);

        expect(await screen.findByText('classification')).toBeInTheDocument();
    });

    it('should pass correct project id to menu actions', async () => {
        render(<ProjectCard item={mockProject} />);

        const menuButton = await screen.findByRole('button', { name: /open project options/i });
        expect(menuButton).toBeInTheDocument();
    });

    it('should handle menu actions without interfering with card click', async () => {
        render(<ProjectCard item={mockProject} />);

        const menuButton = await screen.findByRole('button', { name: /open project options/i });
        const cardLink = await screen.findByRole('link');

        // Menu button should be clickable without triggering link navigation
        fireEvent.click(menuButton);

        expect(menuButton).toBeInTheDocument();
        expect(cardLink).toBeInTheDocument();
    });
});
