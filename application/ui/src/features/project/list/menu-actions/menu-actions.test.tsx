// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { MenuActions } from './menu-actions.component';

describe('MenuActions', () => {
    const projectId = 'test-project-id';
    const projectName = 'Test Project';
    const actionButtonStyle = {};

    it('opens edit dialog when rename menu item is clicked', async () => {
        render(
            <MenuActions
                projectId={projectId}
                projectName={projectName}
                actionButtonStyle={actionButtonStyle}
                projectNames={[]}
            />
        );

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Rename'));

        const inputField = await screen.findByLabelText(/edit project name field/i);
        expect(inputField).toBeVisible();
        expect(inputField).toHaveValue(projectName);
    });

    it('opens delete confirmation dialog when delete menu item is clicked', async () => {
        render(
            <MenuActions
                projectId={projectId}
                projectName={projectName}
                actionButtonStyle={actionButtonStyle}
                projectNames={[]}
            />
        );

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Delete'));

        expect(await screen.findByText(`Are you sure you want to delete project "${projectName}"?`)).toBeVisible();
    });

    it('shows an explanation dialog when enabling pipeline is blocked', async () => {
        render(
            <MenuActions
                projectId={projectId}
                projectName={projectName}
                actionButtonStyle={actionButtonStyle}
                isPipelineRunning={false}
                projectNames={[]}
            />
        );

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Enable pipeline'));

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model, source, and sink before enabling the pipeline.')
        ).toBeVisible();
    });
});
