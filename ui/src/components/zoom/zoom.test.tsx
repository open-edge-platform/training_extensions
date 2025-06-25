import { render, screen } from '@testing-library/react';

import { useContainerSize } from './use-container-size';
import { ZoomProvider } from './zoom';
import { ZoomTransform } from './zoom-transform';

jest.mock('./use-container-size', () => ({
    useContainerSize: jest.fn(),
}));

describe('Zoom', () => {
    it('Scales and translates content so that it fits the screen', () => {
        const screenSize = { width: 100, height: 500 };
        const contentSize = { width: 300, height: 200 };
        const expectedZoom = { translate: { x: -100, y: 150 }, scale: 0.3 };

        jest.mocked(useContainerSize).mockImplementation(() => screenSize);

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
