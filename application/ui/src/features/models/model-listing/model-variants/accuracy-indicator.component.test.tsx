// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import { AccuracyIndicator } from './accuracy-indicator.component';

describe('AccuracyIndicator', () => {
    it('renders pie chart with correct dimensions and centered text', () => {
        const { container } = render(<AccuracyIndicator accuracy={75} />);

        const svg = container.querySelector('svg');
        expect(svg).toBeInTheDocument();
        expect(svg).toHaveAttribute('width', '70');
        expect(svg).toHaveAttribute('height', '50');

        const text = container.querySelector('text');
        expect(text).toHaveTextContent('75%');
        expect(text).toHaveAttribute('text-anchor', 'middle');
        expect(text).toHaveAttribute('x', '50%');
        expect(text).toHaveAttribute('y', '50%');
    });

    it.each([
        { accuracy: 0, desc: '0% accuracy' },
        { accuracy: 25, desc: 'low accuracy (<70)' },
        { accuracy: 50, desc: 'mid-low accuracy' },
        { accuracy: 70, desc: 'medium accuracy threshold' },
        { accuracy: 80, desc: 'medium accuracy (70-89)' },
        { accuracy: 87.5, desc: 'decimal values' },
        { accuracy: 90, desc: 'high accuracy threshold' },
        { accuracy: 95, desc: 'high accuracy (>=90)' },
        { accuracy: 100, desc: '100% accuracy' },
    ])('displays correct percentage for $desc', ({ accuracy }) => {
        const { container } = render(<AccuracyIndicator accuracy={accuracy} />);
        const text = container.querySelector('text');

        expect(text).toHaveTextContent(`${accuracy}%`);
    });
});
