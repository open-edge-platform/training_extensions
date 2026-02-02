// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import type { DatasetGroup } from '../../types';
import { DatasetActions } from './dataset-actions.component';

const mockDataset: DatasetGroup = {
    id: 'dataset-123',
    name: 'Test Dataset',
    createdAt: '10 Jan 2025',
    labelCount: 5,
    imageCount: 100,
    trainingSubsets: {
        training: 70,
        validation: 20,
        testing: 10,
    },
    filesDeleted: false,
};

describe('DatasetActions', () => {
    it('should render menu with all items', async () => {
        render(<DatasetActions dataset={mockDataset} />);

        const menuButton = screen.getByRole('button', { name: 'Dataset actions' });
        expect(menuButton).toBeInTheDocument();

        await userEvent.click(menuButton);

        expect(screen.getByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    });

    it('should open rename dialog when rename action is clicked', async () => {
        render(<DatasetActions dataset={mockDataset} />);

        const menuButton = screen.getByRole('button', { name: 'Dataset actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));

        expect(screen.getByRole('dialog', { name: 'Rename dataset revision' })).toBeInTheDocument();
        expect(screen.getByRole('textbox', { name: /Dataset revision name/ })).toHaveValue('Test Dataset');
    });

    it('should open delete dialog when delete action is clicked', async () => {
        render(<DatasetActions dataset={mockDataset} />);

        const menuButton = screen.getByRole('button', { name: 'Dataset actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

        expect(screen.getByRole('alertdialog', { name: 'Delete dataset revision' })).toBeInTheDocument();
        expect(screen.getByText(/Are you sure you want to delete dataset revision/)).toBeInTheDocument();
    });
});
