// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { useContainerSize } from './use-container-size';
import { ZoomTransform } from './zoom-transform';
import { ZoomProvider } from './zoom.provider';

vi.mock('./use-container-size', () => ({
    useContainerSize: vi.fn(),
}));

describe('Zoom', () => {
    it('Scales and translates content so that it fits the screen', () => {
        const screenSize = { width: 500, height: 500 };
        const contentSize = { width: 500, height: 500 };
        const expectedZoom = 0.9;

        vi.mocked(useContainerSize).mockImplementation(() => screenSize);

        render(
            <ZoomProvider>
                <ZoomTransform target={contentSize}>Content</ZoomTransform>
            </ZoomProvider>
        );

        const transform = screen.getByTestId('zoom-transform');

        expect(transform).toHaveAttribute('data-zoom-value', expectedZoom.toString());
    });
});
