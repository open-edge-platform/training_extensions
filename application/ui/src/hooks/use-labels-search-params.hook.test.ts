// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';

import { renderHook } from '../test-utils/render';
import { encodeFilterSearchParam, useLabelsSearchParams } from './use-labels-search-params.hook';

describe('useLabelsSearchParams', () => {
    it('returns empty selectedLabelIds when no filter param is present', () => {
        const { result } = renderHook(() => useLabelsSearchParams(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });

    it('returns selectedLabelIds from encoded filter param', () => {
        const encoded = encodeFilterSearchParam('id-1,id-2,id-3');

        const { result } = renderHook(() => useLabelsSearchParams(), {
            route: `/projects/123?filters=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual(['id-1', 'id-2', 'id-3']);
    });

    it('returns a single label id', () => {
        const labelName = 'id-1';
        const encoded = encodeFilterSearchParam(labelName);

        const { result } = renderHook(() => useLabelsSearchParams(), {
            route: `/projects/123?filters=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([labelName]);
    });

    it('sets selected label ids in the search params', () => {
        const newLabels = ['id-a', 'id-b'];
        const { result } = renderHook(() => useLabelsSearchParams(), {
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

        const { result } = renderHook(() => useLabelsSearchParams(), {
            route: `/projects/123?filters=${encoded}`,
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([labelName]);

        act(() => {
            result.current.setSelectedLabelIds([]);
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });

    it('returns empty array when filter param is malformed', () => {
        const { result } = renderHook(() => useLabelsSearchParams(), {
            route: '/projects/123?filters=invalid-data',
            path: '/projects/:projectId',
        });

        expect(result.current.selectedLabelIds).toEqual([]);
    });
});
