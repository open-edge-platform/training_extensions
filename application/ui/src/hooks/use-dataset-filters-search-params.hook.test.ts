// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';

import { renderHook } from '../test-utils/render';
import {
    ANNOTATION_STATUS_PARAM,
    encodeFilterSearchParam,
    LABELS_PARAM,
    useDatasetFiltersSearchParams,
} from './use-dataset-filters-search-params.hook';

describe('useDatasetFiltersSearchParams', () => {
    it('returns empty selectedLabelIds when no filter param is present', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });

    it('returns selectedLabelIds from encoded filter param', () => {
        const encoded = encodeFilterSearchParam('id-1,id-2,id-3');

        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${LABELS_PARAM}=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual(['id-1', 'id-2', 'id-3']);
    });

    it('returns a single label id', () => {
        const labelName = 'id-1';
        const encoded = encodeFilterSearchParam(labelName);

        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${LABELS_PARAM}=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([labelName]);
    });

    it('sets selected label ids in the search params', () => {
        const newLabels = ['id-a', 'id-b'];
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        act(() => {
            result.current.setSelectedLabelIds(newLabels);
        });

        expect(result.current.selectedLabelIds).toEqual(newLabels);
    });

    it('clears filter param when setting empty ids', () => {
        const labelName = 'id-1';
        const encoded = encodeFilterSearchParam(labelName);

        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${LABELS_PARAM}=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([labelName]);

        act(() => {
            result.current.setSelectedLabelIds([]);
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });

    it('returns empty array when filter param is malformed', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${LABELS_PARAM}=invalid-data`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });

    it('returns null annotationStatus when no param is present', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        expect(result.current.annotationStatus).toBeNull();
    });

    it.each(['unannotated', 'reviewed', 'to_review'] as const)(
        'returns annotationStatus "%s" from search param',
        (status) => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${ANNOTATION_STATUS_PARAM}=${status}`,
                path: '/projects/:projectId',
            });

            expect(result.current.annotationStatus).toBe(status);
        }
    );

    it('returns null annotationStatus for invalid value', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${ANNOTATION_STATUS_PARAM}=invalid`,
            path: '/projects/:projectId',
        });

        expect(result.current.annotationStatus).toBeNull();
    });

    it('sets annotation status in the search params', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        act(() => {
            result.current.setAnnotationStatus('reviewed');
        });

        expect(result.current.annotationStatus).toBe('reviewed');
    });

    it('clears annotation status when set to null', () => {
        const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
            route: `/projects/123?${ANNOTATION_STATUS_PARAM}=reviewed`,
            path: '/projects/:projectId',
        });

        expect(result.current.annotationStatus).toBe('reviewed');

        act(() => {
            result.current.setAnnotationStatus(null);
        });

        expect(result.current.annotationStatus).toBeNull();
    });
});
