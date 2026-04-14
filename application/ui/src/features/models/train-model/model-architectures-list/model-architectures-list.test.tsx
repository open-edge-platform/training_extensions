// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { render } from 'test-utils/render';

import { ModelArchitecturesList } from './model-architectures-list.component';

const mockOnSelectModelArchitectureId = vi.hoisted(() => vi.fn());

const mockModelArchitectures = [
    getMockedModelArchitecture({ id: 'arch-1', name: 'Alpha Model' }),
    getMockedModelArchitecture({ id: 'arch-2', name: 'Beta Model' }),
    getMockedModelArchitecture({ id: 'arch-3', name: 'Gamma Model' }),
    getMockedModelArchitecture({ id: 'arch-4', name: 'Delta Model' }),
    getMockedModelArchitecture({ id: 'arch-5', name: 'Epsilon Model' }),
];

vi.mock('../train-model-provider.component', () => ({
    useTrainModelState: () => ({
        activeModelArchitectureId: undefined,
        modelArchitectures: mockModelArchitectures,
        selectedModelArchitectureId: null,
        onSelectModelArchitectureId: mockOnSelectModelArchitectureId,
    }),
}));

describe('ModelArchitecturesList', () => {
    beforeEach(() => {
        mockOnSelectModelArchitectureId.mockReset();
    });

    it('does not render the sort control in the default (recommended) view', () => {
        render(<ModelArchitecturesList />);

        expect(screen.queryByText('Sort Models by:')).not.toBeInTheDocument();
    });

    it('renders the sort control after clicking "Show more"', async () => {
        render(<ModelArchitecturesList />);

        fireEvent.click(screen.getByRole('button', { name: 'Show more' }));

        expect(screen.getByRole('button', { name: /Sort Models by:/i })).toBeVisible();
    });

    it('hides the sort control again after clicking "Show less"', async () => {
        render(<ModelArchitecturesList />);

        fireEvent.click(screen.getByRole('button', { name: 'Show more' }));
        expect(screen.getByRole('button', { name: /Sort Models by:/i })).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: 'Show less' }));
        expect(screen.queryByRole('button', { name: /Sort Models by:/i })).not.toBeInTheDocument();
    });
});
