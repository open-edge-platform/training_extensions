// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedLabel } from 'mocks/mock-labels';

import { validateLabelHotkey, validateLabelName } from './label-validation';

describe('validateLabelName', () => {
    const existingLabels = [
        getMockedLabel({ id: 'label-1', name: 'Person' }),
        getMockedLabel({ id: 'label-2', name: 'Car' }),
        getMockedLabel({ id: 'label-3', name: 'Dog' }),
    ];

    it('returns undefined for a unique label name', () => {
        expect(validateLabelName('Cat', existingLabels)).toBeUndefined();
    });

    it('returns error message for duplicate label name', () => {
        expect(validateLabelName('Person', existingLabels)).toBe('That label name already exists');
    });

    it('is case-sensitive when checking duplicates', () => {
        expect(validateLabelName('person', existingLabels)).toBeUndefined();
        expect(validateLabelName('PERSON', existingLabels)).toBeUndefined();
    });

    it('trims whitespace before checking duplicates', () => {
        expect(validateLabelName('  Person  ', existingLabels)).toBe('That label name already exists');
        expect(validateLabelName('  Cat  ', existingLabels)).toBeUndefined();
    });

    it('allows duplicate when excludeId matches the label id', () => {
        expect(validateLabelName('Person', existingLabels, 'label-1')).toBeUndefined();
    });

    it('returns error for duplicate even with excludeId for different label', () => {
        expect(validateLabelName('Person', existingLabels, 'label-2')).toBe('That label name already exists');
    });

    it('returns undefined for empty labels array', () => {
        expect(validateLabelName('Anything', [])).toBeUndefined();
    });

    it('returns undefined for empty name', () => {
        expect(validateLabelName('', existingLabels)).toBeUndefined();
    });

    it('returns undefined for whitespace-only name', () => {
        expect(validateLabelName('   ', existingLabels)).toBeUndefined();
    });
});

describe('validateLabelHotkey', () => {
    it('returns undefined for a unique hotkey', () => {
        expect(validateLabelHotkey('a', ['b', 'c'])).toBeUndefined();
    });

    it('returns error message for duplicate hotkey', () => {
        expect(validateLabelHotkey('a', ['a', 'b'])).toBe('That hotkey is already in use');
    });

    it('is case-insensitive when checking duplicates', () => {
        expect(validateLabelHotkey('A', ['a', 'b'])).toBe('That hotkey is already in use');
        expect(validateLabelHotkey('a', ['A', 'B'])).toBe('That hotkey is already in use');
    });

    it('returns undefined for empty hotkeys array', () => {
        expect(validateLabelHotkey('a', [])).toBeUndefined();
    });

    it('handles modifier key combinations', () => {
        expect(validateLabelHotkey('ctrl+s', ['ctrl+s'])).toBe('That hotkey is already in use');
        expect(validateLabelHotkey('ctrl+s', ['ctrl+a'])).toBeUndefined();
    });
});
