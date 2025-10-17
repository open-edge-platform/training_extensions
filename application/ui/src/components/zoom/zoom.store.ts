// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { clampBetween } from '@geti/smart-tools/utils';
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

import { ZoomState } from './types';
import { getZoomState, ZOOM_STEP_DIVISOR } from './util';

interface ZoomStore extends ZoomState {
    setZoom: (zoom: Partial<ZoomState> | ((state: ZoomState) => Partial<ZoomState>)) => void;
    fitToScreen: () => void;
    onZoomChange: (factor: number, hasAnimation?: boolean) => void;
    zoomToCursor: (newScale: number, cursorX: number, cursorY: number) => void;
    resetZoom: () => void;
}

const initialState: ZoomState = {
    scale: 1.0,
    maxZoomIn: 1,
    hasAnimation: false,
    translate: { x: 0, y: 0 },
    initialCoordinates: { scale: 1.0, x: 0, y: 0 },
};

export const zoomStore = create<ZoomStore>()(
    devtools(
        (set) => ({
            ...initialState,

            setZoom: (zoom) =>
                set(
                    (state) => {
                        const updates = typeof zoom === 'function' ? zoom(state) : zoom;
                        return {
                            ...state,
                            ...updates,
                        };
                    },
                    false,
                    'setZoom'
                ),

            fitToScreen: () =>
                set(
                    (state) => ({
                        hasAnimation: true,
                        scale: state.initialCoordinates.scale,
                        translate: {
                            x: state.initialCoordinates.x,
                            y: state.initialCoordinates.y,
                        },
                    }),
                    false,
                    'fitToScreen'
                ),

            onZoomChange: (factor, hasAnimation = true) =>
                set(
                    (state) => {
                        const step = (state.maxZoomIn - state.initialCoordinates.scale) / ZOOM_STEP_DIVISOR;

                        const newState = getZoomState({
                            newScale: clampBetween(
                                state.initialCoordinates.scale,
                                state.scale + step * factor,
                                state.maxZoomIn
                            ),
                            cursorX: state.initialCoordinates.x,
                            cursorY: state.initialCoordinates.y,
                            initialCoordinates: state.initialCoordinates,
                        })(state);

                        return { ...newState, hasAnimation };
                    },
                    false,
                    'onZoomChange'
                ),

            zoomToCursor: (newScale, cursorX, cursorY) =>
                set(
                    (state) => {
                        const newState = getZoomState({
                            newScale,
                            cursorX,
                            cursorY,
                            initialCoordinates: state.initialCoordinates,
                        })(state);

                        return { ...newState, hasAnimation: false };
                    },
                    false,
                    'zoomToCursor'
                ),

            resetZoom: () =>
                set(
                    {
                        ...initialState,
                    },
                    false,
                    'resetZoom'
                ),
        }),
        { name: 'ZoomStore' }
    )
);

export const useZoom = () => {
    const scale = zoomStore((state) => state.scale);
    const maxZoomIn = zoomStore((state) => state.maxZoomIn);
    const hasAnimation = zoomStore((state) => state.hasAnimation);
    const translate = zoomStore((state) => state.translate);
    const initialCoordinates = zoomStore((state) => state.initialCoordinates);

    return { scale, maxZoomIn, hasAnimation, translate, initialCoordinates };
};

export const useSetZoom = () => {
    const setZoom = zoomStore((state) => state.setZoom);
    const fitToScreen = zoomStore((state) => state.fitToScreen);
    const onZoomChange = zoomStore((state) => state.onZoomChange);
    const zoomToCursor = zoomStore((state) => state.zoomToCursor);
    const resetZoom = zoomStore((state) => state.resetZoom);

    return { setZoom, fitToScreen, onZoomChange, zoomToCursor, resetZoom };
};
