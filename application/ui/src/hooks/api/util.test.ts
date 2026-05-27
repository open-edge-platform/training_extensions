// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isInvalidStagedFile } from './util';

describe('isInvalidStagedFile', () => {
    it('returns true when detail starts with "Staged dataset" and ends with "not found"', () => {
        expect(isInvalidStagedFile({ detail: 'Staged dataset with ID abc-123 not found' })).toBe(true);
        expect(isInvalidStagedFile({ detail: 'Staged dataset not found' })).toBe(true);
        expect(isInvalidStagedFile({ detail: 'staged dataset with ID abc-123 not found.' })).toBe(true);
        expect(isInvalidStagedFile({ detail: ' Staged dataset not found ' })).toBe(true);
    });

    it('returns false when detail does not start with "Staged dataset"', () => {
        expect(isInvalidStagedFile({ detail: 'Resource not found' })).toBe(false);
        expect(isInvalidStagedFile({ detail: 'Staged file not found' })).toBe(false);
    });

    it('returns false when detail does not contain "not found"', () => {
        expect(isInvalidStagedFile({ detail: 'Staged dataset with ID abc-123 missing' })).toBe(false);
        expect(isInvalidStagedFile({ detail: 'Staged dataset' })).toBe(false);
    });
});
