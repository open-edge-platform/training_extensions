// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { ThreeSectionRange } from './three-section-range.component';

describe('ThreeSectionRange', () => {
    it('renders label and displays percentage values correctly', () => {
        render(<ThreeSectionRange trainingValue={70} validationValue={20} testingValue={10} />);

        expect(screen.getByText('TRAINING SUBSETS')).toBeInTheDocument();
        expect(screen.getByText('70% / 20% / 10%')).toBeInTheDocument();
    });

    it.each([
        { training: 33, validation: 33, testing: 34, expected: '33% / 33% / 34%', desc: 'equal distribution' },
        { training: 0, validation: 50, testing: 50, expected: '0% / 50% / 50%', desc: 'zero training' },
        { training: 80, validation: 0, testing: 20, expected: '80% / 0% / 20%', desc: 'zero validation' },
        { training: 70, validation: 30, testing: 0, expected: '70% / 30% / 0%', desc: 'zero testing' },
        { training: 0, validation: 0, testing: 0, expected: '0% / 0% / 0%', desc: 'all zeros' },
        { training: 100, validation: 0, testing: 0, expected: '100% / 0% / 0%', desc: '100% training' },
        { training: 66.6, validation: 16.7, testing: 16.7, expected: '66.6% / 16.7% / 16.7%', desc: 'decimal values' },
    ])('handles various distributions: $desc', ({ training, validation, testing, expected }) => {
        render(<ThreeSectionRange trainingValue={training} validationValue={validation} testingValue={testing} />);

        expect(screen.getByText(expected)).toBeInTheDocument();
    });
});
