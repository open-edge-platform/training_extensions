// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModelArchitecture } from '../../../../../mocks/mock-model';
import { getRecommendedModelArchitecturesWithActiveArchitecture } from './utils';

describe('getRecommendedModelArchitecturesWithActiveArchitecture', () => {
    it('returns recommended architectures when performanceCategory is defined', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1', performanceCategory: 'balance' }),
            getMockedModelArchitecture({ id: 'arch-2', performanceCategory: 'speed' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, undefined);

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });

    it('returns top 3 architectures when no performanceCategory is defined', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
            getMockedModelArchitecture({ id: 'arch-4' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, undefined);

        expect(result).toHaveLength(3);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
        expect(result[2].id).toBe('arch-3');
    });

    it('returns recommended architectures when active architecture is already in recommended', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1', performanceCategory: 'balance' }),
            getMockedModelArchitecture({ id: 'arch-2', performanceCategory: 'speed' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, 'arch-1');

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });

    it('prepends active architecture when it is not in recommended list', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1', performanceCategory: 'balance' }),
            getMockedModelArchitecture({ id: 'arch-2', performanceCategory: 'speed' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
            getMockedModelArchitecture({ id: 'arch-4' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, 'arch-4');

        expect(result).toHaveLength(3);
        expect(result[0].id).toBe('arch-4');
        expect(result[1].id).toBe('arch-1');
        expect(result[2].id).toBe('arch-2');
    });

    it('returns recommended architectures when active architecture ID does not exist', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1', performanceCategory: 'balance' }),
            getMockedModelArchitecture({ id: 'arch-2', performanceCategory: 'speed' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, 'non-existent');

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });

    it('handles empty model architectures array', () => {
        const result = getRecommendedModelArchitecturesWithActiveArchitecture([], undefined);

        expect(result).toHaveLength(0);
    });

    it('handles model architectures with less than 3 items and no performanceCategory', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, undefined);

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });

    it('prepends active architecture when only using top 3 fallback', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
            getMockedModelArchitecture({ id: 'arch-4' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, 'arch-4');

        expect(result).toHaveLength(4);
        expect(result[0].id).toBe('arch-4');
        expect(result[1].id).toBe('arch-1');
        expect(result[2].id).toBe('arch-2');
        expect(result[3].id).toBe('arch-3');
    });

    it('handles active architecture in top 3 when no performanceCategory', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
            getMockedModelArchitecture({ id: 'arch-4' }),
        ];

        const result = getRecommendedModelArchitecturesWithActiveArchitecture(modelArchitectures, 'arch-2');

        expect(result).toHaveLength(3);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
        expect(result[2].id).toBe('arch-3');
    });
});
