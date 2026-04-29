// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getUniqueName } from './utils';

describe('getUniqueName', () => {
    it('returns the base name when no existing names', () => {
        expect(getUniqueName('USB camera source', [])).toBe('USB camera source');
    });

    it('returns the base name when it does not conflict with existing names', () => {
        expect(getUniqueName('USB camera source', ['IP camera source', 'Video file source'])).toBe('USB camera source');
    });

    it('returns "baseName - 1" when base name already exists', () => {
        expect(getUniqueName('USB camera source', ['USB camera source'])).toBe('USB camera source - 1');
    });

    it('returns "baseName - 2" when both base name and "baseName - 1" already exist', () => {
        expect(getUniqueName('USB camera source', ['USB camera source', 'USB camera source - 1'])).toBe(
            'USB camera source - 2'
        );
    });

    it('skips to the next available counter even when some are missing', () => {
        expect(
            getUniqueName('USB camera source', ['USB camera source', 'USB camera source - 1', 'USB camera source - 2'])
        ).toBe('USB camera source - 3');
    });
});
