// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toggleMultipleSelection } from './util';

describe('toggleMultipleSelection', () => {
    const items = ['a', 'b', 'c'];

    it('returns empty set if selectedItems is "all"', () => {
        const result = toggleMultipleSelection(items)('all');
        expect(result).toEqual(new Set());
    });

    it('should select all items if selectedItems is empty set', () => {
        const result = toggleMultipleSelection(items)(new Set());
        expect(result).toEqual(new Set(items));
    });

    it('should select all items if some items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(['a']));
        expect(result).toEqual(new Set(items));
    });

    it('should deselect all items if all items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(items));
        expect(result).toEqual(new Set());
    });

    it('should select all items if more than one but not all items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(['a', 'b']));
        expect(result).toEqual(new Set(items));
    });
});
