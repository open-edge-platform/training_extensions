// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { render } from 'test-utils/render';

import { ModelActions } from './model-actions.component';

const mockModel = getMockedModel({
    id: 'model-123',
    name: 'Test Model',
    architecture: 'YOLOX',
    size: 1024,
    training_info: {
        status: 'successful',
        label_schema_revision: {
            labels: [],
        },
        start_time: '2025-01-10T10:00:00.000000+00:00',
        end_time: '2025-01-10T12:30:00.000000+00:00',
        dataset_revision_id: 'dataset-123',
    },
});

describe('ModelActions', () => {
    it('should render menu with all items', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        expect(menuButton).toBeInTheDocument();

        await userEvent.click(menuButton);

        expect(screen.getByRole('menuitem', { name: 'Set as active' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    });

    it('should open rename dialog when rename action is clicked', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));

        expect(screen.getByRole('dialog', { name: 'Rename model' })).toBeInTheDocument();
        expect(screen.getByRole('textbox', { name: /Model name/ })).toHaveValue('Test Model');
    });

    it('should open delete dialog when delete action is clicked', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

        expect(screen.getByRole('alertdialog', { name: 'Delete model' })).toBeInTheDocument();
        expect(screen.getByText(/Are you sure you want to delete/)).toBeInTheDocument();
    });
});
