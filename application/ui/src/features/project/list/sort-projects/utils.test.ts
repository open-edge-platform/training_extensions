// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedProject } from 'mocks/mock-project';

import { SORT_BY_HANDLERS } from './utils';

describe('SORT_BY_HANDLERS', () => {
    const apple = getMockedProject({ name: 'apple', created_at: '2024-01-01T00:00:00Z' });
    const apricot = getMockedProject({ name: 'Apricot', created_at: '2024-05-01T00:00:00Z' });
    const banana = getMockedProject({ name: 'Banana', created_at: '2024-03-01T00:00:00Z' });
    const cherry = getMockedProject({ name: 'cherry', created_at: '2024-02-01T00:00:00Z' });
    const projects = [cherry, apricot, banana, apple];

    describe('Name ascending', () => {
        it('sorts projects A to Z case-insensitively', () => {
            const result = SORT_BY_HANDLERS['name-ascending'](projects);
            expect(result.map((p) => p.name)).toEqual([apple.name, apricot.name, banana.name, cherry.name]);
        });

        it('places lowercase letters before uppercase of a later letter', () => {
            const result = SORT_BY_HANDLERS['name-ascending']([apricot, apple]);
            expect(result[0].name).toBe(apple.name);
            expect(result[1].name).toBe(apricot.name);
        });
    });

    describe('Name descending', () => {
        it('sorts projects Z to A case-insensitively', () => {
            const result = SORT_BY_HANDLERS['name-descending'](projects);
            expect(result.map((p) => p.name)).toEqual([cherry.name, banana.name, apricot.name, apple.name]);
        });

        it('places uppercase of a later letter before lowercase of an earlier letter', () => {
            const result = SORT_BY_HANDLERS['name-descending']([apple, apricot]);
            expect(result[0].name).toBe(apricot.name);
            expect(result[1].name).toBe(apple.name);
        });
    });

    describe('Creation date ascending', () => {
        it('sorts projects with oldest created_at first', () => {
            const result = SORT_BY_HANDLERS['createdAt-ascending'](projects);
            expect(result.map((p) => p.name)).toEqual([apple.name, cherry.name, banana.name, apricot.name]);
        });
    });

    describe('Creation date descending', () => {
        it('sorts projects with newest created_at first', () => {
            const result = SORT_BY_HANDLERS['createdAt-descending'](projects);
            expect(result.map((p) => p.name)).toEqual([apricot.name, banana.name, cherry.name, apple.name]);
        });
    });
});
