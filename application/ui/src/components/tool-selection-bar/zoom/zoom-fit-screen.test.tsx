// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from 'test-utils/render';
import { describe, expect, it, vi } from 'vitest';

import { ZoomFitScreen } from './zoom-fit-screen.component';

const mockedSetZoom = vi.fn();
vi.mock(import('../../../components/zoom/zoom.provider'), async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual,
        useSetZoom: () => mockedSetZoom,
    };
});

describe('ZoomFitScreen', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('calls onFitScreen when button is clicked', () => {
        render(<ZoomFitScreen />);

        fireEvent.click(screen.getByRole('button', { name: /fit to screen/i }));

        expect(mockedSetZoom).toHaveBeenCalledTimes(1);
        const setZoomCall = mockedSetZoom.mock.calls[0][0];

        const mockPrev = {
            scale: 2,
            translate: { x: 100, y: 200 },
            initialCoordinates: { scale: 1, x: 0, y: 0 },
        };

        expect(setZoomCall(mockPrev)).toEqual({
            ...mockPrev,
            scale: mockPrev.initialCoordinates.scale,
            translate: { x: mockPrev.initialCoordinates.x, y: mockPrev.initialCoordinates.y },
        });
    });
});
