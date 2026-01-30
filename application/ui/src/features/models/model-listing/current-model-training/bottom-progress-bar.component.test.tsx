// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { BottomProgressBar } from './bottom-progress-bar.component';

describe('BottomProgressBar', () => {
    it.each([
        { progress: 0, desc: '0% progress' },
        { progress: 25, desc: '25% progress' },
        { progress: 50, desc: '50% progress' },
        { progress: 75, desc: '75% progress' },
        { progress: 100, desc: '100% progress' },
        { progress: 33.33, desc: 'decimal progress' },
        { progress: -10, desc: 'negative value' },
        { progress: 150, desc: 'over 100%' },
    ])('handles various progress values: $desc', ({ progress }) => {
        const { container } = render(
            <BottomProgressBar progress={progress}>
                <div>Content</div>
            </BottomProgressBar>
        );

        expect(screen.getByText('Content')).toBeInTheDocument();
        expect(container.firstChild).toBeInTheDocument();
    });
});
