;

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { stringify } from 'zipson/lib';



import { renderHook } from '../test-utils/render';
import { ANNOTATION_STATUS_PARAM, encodeToBinary, END_DATE_PARAM, LABELS_PARAM, START_DATE_PARAM, useDatasetFiltersSearchParams } from './use-dataset-filters-search-params.hook';





;
















;















describe('useDatasetFiltersSearchParams', () => {
    describe('labels filter', () => {
        it('returns empty selectedLabelIds when no filter param is present', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: '/projects/123',
                path: '/projects/:projectId',
            });

            expect(result.current.selectedLabelIds).toEqual([]);
        });

        it('returns selectedLabelIds from encoded filter param', () => {
            const encoded = encodeToBinary(stringify('id-1,id-2,id-3'));

            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${LABELS_PARAM}=${encoded}`,
                path: '/projects/:projectId',
            });

            expect(result.current.selectedLabelIds).toEqual(['id-1', 'id-2', 'id-3']);
        });

        it('returns a single label id', () => {
            const labelName = 'id-1';
            const encoded = encodeToBinary(stringify(labelName));

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
            const encoded = encodeToBinary(stringify(labelName));

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
    });

    describe('annotationStatus filter', () => {
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
                result.current.setAnnotationStatus('with_annotations');
            });

            expect(result.current.annotationStatus).toBe('with_annotations');
        });

        it('clears annotation status when set to null', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${ANNOTATION_STATUS_PARAM}=with_annotations`,
                path: '/projects/:projectId',
            });

            expect(result.current.annotationStatus).toBe('with_annotations');

            act(() => {
                result.current.setAnnotationStatus(null);
            });

            expect(result.current.annotationStatus).toBeNull();
        });
    });

    describe('date filters', () => {
        it('returns null startDate and endDate when no params are present', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: '/projects/123',
                path: '/projects/:projectId',
            });

            expect(result.current.startDate).toBeNull();
            expect(result.current.endDate).toBeNull();
        });

        it('returns startDate from search param', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${START_DATE_PARAM}=2026-01-01`,
                path: '/projects/:projectId',
            });

            expect(result.current.startDate).toBe('2026-01-01');
        });

        it('returns endDate from search param', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${END_DATE_PARAM}=2026-12-31`,
                path: '/projects/:projectId',
            });

            expect(result.current.endDate).toBe('2026-12-31');
        });

        it('sets startDate in the search params', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: '/projects/123',
                path: '/projects/:projectId',
            });

            act(() => {
                result.current.setStartDate('2026-03-15');
            });

            expect(result.current.startDate).toBe('2026-03-15');
        });

        it('sets endDate in the search params', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: '/projects/123',
                path: '/projects/:projectId',
            });

            act(() => {
                result.current.setEndDate('2026-06-30');
            });

            expect(result.current.endDate).toBe('2026-06-30');
        });

        it('clears startDate when set to null', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${START_DATE_PARAM}=2026-01-01`,
                path: '/projects/:projectId',
            });

            expect(result.current.startDate).toBe('2026-01-01');

            act(() => {
                result.current.setStartDate(null);
            });

            expect(result.current.startDate).toBeNull();
        });

        it('clears endDate when set to null', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${END_DATE_PARAM}=2026-12-31`,
                path: '/projects/:projectId',
            });

            expect(result.current.endDate).toBe('2026-12-31');

            act(() => {
                result.current.setEndDate(null);
            });

            expect(result.current.endDate).toBeNull();
        });

        it('returns both startDate and endDate when both params are present', () => {
            const { result } = renderHook(() => useDatasetFiltersSearchParams(), {
                route: `/projects/123?${START_DATE_PARAM}=2026-01-01&${END_DATE_PARAM}=2026-12-31`,
                path: '/projects/:projectId',
            });

            expect(result.current.startDate).toBe('2026-01-01');
            expect(result.current.endDate).toBe('2026-12-31');
        });
    });
});
