// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { render, screen } from '@testing-library/react';

import { ThinProgressBar, ThinProgressBarProps } from './thin-progress-bar.component';

describe('UploadStatusProgressBar', () => {
    it('renders an element with the correct styles based on props', () => {
        const testProps: ThinProgressBarProps = {
            progress: 0,
            size: 'size-50',
            color: 'blue-400',
            customColor: '',
            trackColor: 'gray-400',
        };

        render(<ThinProgressBar {...testProps} />);

        expect(screen.getByTestId('thin-progress-bar')).toHaveStyle({
            height: testProps.size,
            backgroundColor: testProps.color,
            width: `${testProps.progress}%`,
        });
    });
});
