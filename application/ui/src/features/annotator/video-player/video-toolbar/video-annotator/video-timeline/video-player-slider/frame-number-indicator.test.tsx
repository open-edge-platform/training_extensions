// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { FrameNumberIndicator } from './frame-number-indicator.component';

describe('FrameNumberIndicator', () => {
    it('renders the frame number with an "f" suffix', () => {
        render(<FrameNumberIndicator frameNumber={42} />);

        expect(screen.getByText('42f')).toBeInTheDocument();
    });

    it('renders zero frames', () => {
        render(<FrameNumberIndicator frameNumber={0} />);

        expect(screen.getByText('0f')).toBeInTheDocument();
    });

    it('renders large frame numbers in compact notation', () => {
        render(<FrameNumberIndicator frameNumber={1000} />);

        expect(screen.getByText('1Kf')).toBeInTheDocument();
    });
});
