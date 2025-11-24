// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from 'test-utils/render';

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
        const expectedZoom = { translate: { x: 0, y: 0 }, scale: 1 };

        vi.mocked(useContainerSize).mockImplementation(() => screenSize);

        render(
            <ZoomProvider>
                <ZoomTransform target={contentSize}>Content</ZoomTransform>
            </ZoomProvider>
        );

        const transform = screen.getByTestId('zoom-transform');

        expect(transform).toHaveStyle({
            transform: `translate(${expectedZoom.translate.x}px, ${expectedZoom.translate.y}px) scale(${expectedZoom.scale})`,
        });
    });
});
