// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ThreeSectionRange } from './three-section-range.component';

const getDisplayedPercentages = (): number[] => {
    const text = screen.getByText(/Training\s+\d+%\s*\/\s*Validation\s+\d+%\s*\/\s*Test\s+\d+%/).textContent ?? '';
    return Array.from(text.matchAll(/(\d+)%/g)).map((match) => Number(match[1]));
};

describe('ThreeSectionRange', () => {
    it('labels each percentage with the corresponding subset name', () => {
        render(<ThreeSectionRange trainingValue={70} validationValue={20} testingValue={10} />);

        expect(screen.getByText('Training 70% / Validation 20% / Test 10%')).toBeInTheDocument();
    });

    it('exposes each colored segment via an aria-label that includes the subset name and percentage', () => {
        render(<ThreeSectionRange trainingValue={70} validationValue={20} testingValue={10} />);

        expect(screen.getByLabelText('Training: 70%')).toBeInTheDocument();
        expect(screen.getByLabelText('Validation: 20%')).toBeInTheDocument();
        expect(screen.getByLabelText('Test: 10%')).toBeInTheDocument();
    });

    it('renders a zero percentage for every subset when there are no items', () => {
        render(<ThreeSectionRange trainingValue={0} validationValue={0} testingValue={0} />);

        expect(screen.getByText('Training 0% / Validation 0% / Test 0%')).toBeInTheDocument();
    });

    it('rounds percentages so that they always sum to 100% (largest remainder method)', () => {
        // Exact percentages: 65.4 / 23.6 / 11.7 (sum 100.7 -> independent rounding gives 65/24/12 = 101)
        render(<ThreeSectionRange trainingValue={654} validationValue={236} testingValue={117} />);

        const displayed = getDisplayedPercentages();
        expect(displayed.reduce((sum, value) => sum + value, 0)).toBe(100);
    });

    it('rounds percentages so that they always sum to 100% when the floored sum is below 100', () => {
        // Exact percentages: 33.33... / 33.33... / 33.33... -> floors to 33/33/33 (sum 99)
        render(<ThreeSectionRange trainingValue={1} validationValue={1} testingValue={1} />);

        const displayed = getDisplayedPercentages();
        expect(displayed.reduce((sum, value) => sum + value, 0)).toBe(100);
    });
});
