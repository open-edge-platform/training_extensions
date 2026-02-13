// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { ThinProgressBar, ThinProgressBarProps } from './thin-progress-bar.component';

describe('ThinProgressBar', () => {
    it('renders an element with the correct styles based on props', () => {
        const testProps: ThinProgressBarProps = {
            progress: 0,
            size: 'size-50',
            color: 'blue-400',
            customColor: '',
            trackColor: 'gray-400',
        };

        render(<ThinProgressBar {...testProps} />);

        expect(screen.getByTestId('thin-progress-bar')).toHaveStyle({ width: `${testProps.progress}%` });
    });
});
