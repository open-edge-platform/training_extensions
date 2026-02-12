// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

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

    it('opens delete confirmation dialog when delete menu item is clicked', async () => {
        render(<MenuActions projectId={projectId} projectName={projectName} actionButtonStyle={actionButtonStyle} />);

        fireEvent.click(screen.getByLabelText(/open project options/i));
        fireEvent.click(await screen.findByText('Delete'));

        expect(await screen.findByText(`Are you sure you want to delete project "${projectName}"?`)).toBeVisible();
    });
});
