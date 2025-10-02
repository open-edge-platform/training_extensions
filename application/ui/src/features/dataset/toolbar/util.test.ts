// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';

import { MediaState } from '../../../routes/dataset/provider';
import { toggleMultipleSelection, updateSelectedKeysTo } from './util';

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
describe('updateSelectedKeysTo', () => {
    it('returns a new map with updated annotationState for each selected key', () => {
        const selectedKeys = new Set(['1', '2']);
        const initialMap: MediaState = new Map([
            ['1', 'accepted'],
            ['2', 'accepted'],
        ]);

        const result = updateSelectedKeysTo(selectedKeys, 'accepted')(initialMap);

        expect(result.get('1')).toBe('accepted');
        expect(result.get('2')).toBe('accepted');

        expect(result).not.toBe(initialMap);
    });

    it('returns a new map with all entries if selectedKeys is "all"', () => {
        const selectedKeys = 'all';
        const initialMap: MediaState = new Map([
            ['1', 'accepted'],
            ['2', 'accepted'],
        ]);

        const result = updateSelectedKeysTo(selectedKeys, 'rejected')(initialMap);

        expect(result).not.toBe(initialMap);
        expect(result).toEqual(initialMap);
    });
});
