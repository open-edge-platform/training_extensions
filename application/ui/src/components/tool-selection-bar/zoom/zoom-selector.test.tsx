// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from 'test-utils/render';
import { describe, expect, it, vi } from 'vitest';

import { useZoom } from '../../../components/zoom/zoom.provider';
import { ZoomSelector } from './zoom-selector.component';

const mockedSetZoom = vi.fn();
vi.mock(import('../../../components/zoom/zoom.provider'), async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual,
        useZoom: vi.fn(),
        useSetZoom: () => mockedSetZoom,
    };
});

describe('ZoomSelector', () => {
    it('displays the correct zoom value', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.21,
            maxZoomIn: 0,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        expect(screen.getByText('21.0%')).toBeVisible();
    });

    it('calls setZoom when zoom in button is clicked', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.5,
            maxZoomIn: 2,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        const zoomInButton = screen.getByRole('button', { name: /zoom in/i });
        fireEvent.click(zoomInButton);
        expect(mockedSetZoom).toHaveBeenCalled();
    });

    it('calls setZoom when zoom out button is clicked', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 1,
            maxZoomIn: 2,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
        fireEvent.click(zoomOutButton);
        expect(mockedSetZoom).toHaveBeenCalled();
    });

    it('disables zoom in button at max zoom in', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 2,
            maxZoomIn: 2,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);

        expect(screen.getByRole('button', { name: /zoom in/i })).toBeDisabled();
    });

    it('disables zoom out button at max zoom out', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.1,
            maxZoomIn: 2,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0.1, x: 0, y: 0 },
        });

        render(<ZoomSelector />);

        expect(screen.getByRole('button', { name: /zoom out/i })).toBeDisabled();
    });
});
