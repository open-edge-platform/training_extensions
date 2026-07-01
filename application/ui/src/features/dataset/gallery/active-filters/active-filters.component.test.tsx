// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import { ActiveFilters } from './active-filters.component';

vi.mock('hooks/use-project-labels.hook', () => ({
    useProjectLabels: vi.fn(),
}));

vi.mock('hooks/use-dataset-filters-search-params.hook', () => ({
    useDatasetFiltersSearchParams: vi.fn(),
}));

const mockLabels = [getMockedLabel({ id: 'label-1', name: 'Cat' }), getMockedLabel({ id: 'label-2', name: 'Dog' })];

const mockSetSelectedLabelIds = vi.fn();
const mockSetAnnotationStatus = vi.fn();
const mockSetStartDate = vi.fn();
const mockSetEndDate = vi.fn();

const mockUseDatasetFiltersSearchParams = (overrides?: Partial<ReturnType<typeof useDatasetFiltersSearchParams>>) => {
    vi.mocked(useDatasetFiltersSearchParams).mockReturnValue({
        selectedLabelIds: [],
        setSelectedLabelIds: mockSetSelectedLabelIds,
        annotationStatus: null,
        setAnnotationStatus: mockSetAnnotationStatus,
        startDate: null,
        setStartDate: mockSetStartDate,
        endDate: null,
        setEndDate: mockSetEndDate,
        ...overrides,
    });
};

describe('ActiveFilters', () => {
    beforeEach(() => {
        vi.mocked(useProjectLabels).mockReturnValue(mockLabels);
    });

    it('renders nothing when there are no active filters', () => {
        mockUseDatasetFiltersSearchParams();

        render(<ActiveFilters />);

        expect(screen.queryByLabelText('Active filters')).not.toBeInTheDocument();
    });

    it('renders a chip for each selected label', () => {
        mockUseDatasetFiltersSearchParams({ selectedLabelIds: ['label-1', 'label-2'] });

        render(<ActiveFilters />);

        expect(screen.getByText('Cat')).toBeVisible();
        expect(screen.getByText('Dog')).toBeVisible();
    });

    it('renders a chip for the annotation status filter', () => {
        mockUseDatasetFiltersSearchParams({ annotationStatus: 'with_annotations' });

        render(<ActiveFilters />);

        expect(screen.getByText('Media with annotations')).toBeVisible();
    });

    it('renders a chip for the missing annotations status filter', () => {
        mockUseDatasetFiltersSearchParams({ annotationStatus: 'missing_annotations' });

        render(<ActiveFilters />);

        expect(screen.getByText('Media with missing annotations')).toBeVisible();
    });

    it('renders chips for the start and end date filters', () => {
        mockUseDatasetFiltersSearchParams({ startDate: '2026-01-01', endDate: '2026-01-31' });

        render(<ActiveFilters />);

        expect(screen.getByText('from 01/01/2026')).toBeVisible();
        expect(screen.getByText('to 31/01/2026')).toBeVisible();
    });

    it('removes only the clicked label when its chip is closed', () => {
        mockUseDatasetFiltersSearchParams({ selectedLabelIds: ['label-1', 'label-2'] });

        render(<ActiveFilters />);

        const catChipContainer = screen.getByText('Cat').closest('div');
        const closeIcon = catChipContainer?.querySelector('svg');

        expect(closeIcon).not.toBeNull();

        fireEvent.click(closeIcon as SVGElement);

        expect(mockSetSelectedLabelIds).toHaveBeenCalledWith(['label-2']);
    });

    it('clears all filters when "Clear all" is pressed', () => {
        mockUseDatasetFiltersSearchParams({
            selectedLabelIds: ['label-1'],
            annotationStatus: 'with_annotations',
            startDate: '2026-01-01',
            endDate: '2026-01-31',
        });

        render(<ActiveFilters />);

        fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));

        expect(mockSetSelectedLabelIds).toHaveBeenCalledWith([]);
        expect(mockSetAnnotationStatus).toHaveBeenCalledWith(null);
        expect(mockSetStartDate).toHaveBeenCalledWith(null);
        expect(mockSetEndDate).toHaveBeenCalledWith(null);
    });
});
