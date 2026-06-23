// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { render } from 'test-utils/render';

import { ModelArchitecturesList } from './model-architectures-list.component';

const mockOnSelectModelArchitectureId = vi.hoisted(() => vi.fn());
const mockSelectedModelArchitectureId = vi.hoisted(() => ({ value: null as string | null }));
const mockShowMore = vi.hoisted(() => ({ value: false as boolean }));
const mockSetShowMore = vi.hoisted(() =>
    vi.fn((val: boolean) => {
        mockShowMore.value = val;
    })
);

const mockModelArchitectures = [
    getMockedModelArchitecture({ id: 'arch-1', name: 'Alpha Model' }),
    getMockedModelArchitecture({ id: 'arch-2', name: 'Beta Model' }),
    getMockedModelArchitecture({ id: 'arch-3', name: 'Gamma Model' }),
    getMockedModelArchitecture({ id: 'arch-4', name: 'Delta Model' }),
    getMockedModelArchitecture({ id: 'arch-5', name: 'Epsilon Model' }),
];

vi.mock('../train-model-provider.component', () => ({
    useTrainModelState: () => ({
        modelArchitectures: mockModelArchitectures,
        selectedModelArchitectureId: mockSelectedModelArchitectureId.value,
        onSelectModelArchitectureId: mockOnSelectModelArchitectureId,
        showMoreModelArchitectures: mockShowMore.value,
        setShowMoreModelArchitectures: mockSetShowMore,
    }),
}));

describe('ModelArchitecturesList', () => {
    beforeEach(() => {
        mockOnSelectModelArchitectureId.mockReset();
        mockSelectedModelArchitectureId.value = null;
        mockShowMore.value = false;
        mockSetShowMore.mockClear();
    });

    it('does not render the sort control in the default (recommended) view', () => {
        render(<ModelArchitecturesList />);

        expect(screen.queryByText('Sort Models by:')).not.toBeInTheDocument();
    });

    it('calls setShowMore with true when clicking "Show more"', async () => {
        render(<ModelArchitecturesList />);

        fireEvent.click(screen.getByRole('button', { name: 'Show more' }));

        expect(mockSetShowMore).toHaveBeenCalledWith(true);
    });

    it('calls setShowMore with false when clicking "Show less"', async () => {
        mockShowMore.value = true;
        render(<ModelArchitecturesList />);

        fireEvent.click(screen.getByRole('button', { name: 'Show less' }));

        expect(mockSetShowMore).toHaveBeenCalledWith(false);
    });

    it('calls setShowMore to true on mount if a non-default architecture is already selected', () => {
        mockSelectedModelArchitectureId.value = mockModelArchitectures[mockModelArchitectures.length - 1].id;
        render(<ModelArchitecturesList />);

        expect(mockSetShowMore).toHaveBeenCalledWith(true);
    });
});
