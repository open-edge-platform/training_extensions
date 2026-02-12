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

    describe('unique names', () => {
        it.each([
            ['unique name', 'Cat'],
            ['empty labels array', 'Anything'],
            ['empty name', ''],
            ['whitespace-only name', '   '],
        ])('returns undefined for %s', (_, name) => {
            const labels = name === 'Anything' ? [] : existingLabels;
            expect(validateLabelName(name, labels, undefined)).toBeUndefined();
        });
    });

    describe('duplicate detection', () => {
        it('returns error for duplicate name', () => {
            expect(validateLabelName('Person', existingLabels)).toBe('That label name already exists');
        });

        it('trims whitespace before checking', () => {
            expect(validateLabelName('  Person  ', existingLabels)).toBe('That label name already exists');
        });

        it('is case-sensitive', () => {
            expect(validateLabelName('person', existingLabels)).toBeUndefined();
            expect(validateLabelName('PERSON', existingLabels)).toBeUndefined();
        });
    });

    describe('excludeId', () => {
        it('allows same name when excludeId matches label id', () => {
            expect(validateLabelName('Person', existingLabels, 'label-1')).toBeUndefined();
        });

        it('still detects duplicate when excludeId is different label', () => {
            expect(validateLabelName('Person', existingLabels, 'label-2')).toBe('That label name already exists');
        });
    });
});

describe('validateLabelHotkey', () => {
    it.each([
        ['unique hotkey', 'a', ['b', 'c']],
        ['empty array', 'a', []],
        ['unique with modifiers', 'ctrl+s', ['ctrl+a']],
    ])('returns undefined for %s', (_, hotkey, existing) => {
        expect(validateLabelHotkey(hotkey, existing)).toBeUndefined();
    });

    it.each([
        ['exact match', 'a', ['a', 'b']],
        ['uppercase input', 'A', ['a', 'b']],
        ['lowercase input', 'a', ['A', 'B']],
        ['modifier combination', 'ctrl+s', ['ctrl+s']],
    ])('returns error for duplicate: %s', (_, hotkey, existing) => {
        expect(validateLabelHotkey(hotkey, existing)).toBe('That hotkey is already in use');
    });
});
