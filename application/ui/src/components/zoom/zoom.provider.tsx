// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { clampBetween } from '@geti/smart-tools/utils';

import { ZoomState } from './types';
import { getZoomState, ZOOM_STEP_DIVISOR } from './util';

const Zoom = createContext<ZoomState>({
    scale: 1.0,
    maxZoomIn: 1,
    hasAnimation: false,
    translate: { x: 0, y: 0 },
    initialCoordinates: { scale: 1.0, x: 0, y: 0 },
});

type SetZoomContextProps = {
    setZoom: Dispatch<SetStateAction<ZoomState>>;
    fitToScreen: () => void;
    onZoomChange: (factor: number) => void;
};

const SetZoom = createContext<SetZoomContextProps | null>(null);

export const useZoom = () => {
    const context = useContext(Zoom);

    if (!context) {
        throw new Error('useZoom must be used within "Zoom.Provider"');
    }

    return context;
};

export const useSetZoom = () => {
    const context = useContext(SetZoom);

    if (!context) {
        throw new Error('useSetZoom must be used within "SetZoom.Provider"');
    }

    return context;
};

export const ZoomProvider = ({ children }: { children: ReactNode }) => {
    // 1. Add translate restrictions - min max
    const [zoom, setZoom] = useState<ZoomState>({
        scale: 1.0,
        maxZoomIn: 1,
        hasAnimation: false,
        translate: { x: 0, y: 0 },
        initialCoordinates: { scale: 1.0, x: 0, y: 0 },
    });

    const fitToScreen = () => {
        setZoom((prev) => ({
            ...prev,
            hasAnimation: true,
            scale: prev.initialCoordinates.scale,
            translate: { x: prev.initialCoordinates.x, y: prev.initialCoordinates.y },
        }));
    };

    const onZoomChange = (factor: number, hasAnimation = true) => {
        setZoom((prev) => {
            const step = (prev.maxZoomIn - prev.initialCoordinates.scale) / ZOOM_STEP_DIVISOR;

            const newState = getZoomState({
                newScale: clampBetween(prev.initialCoordinates.scale, prev.scale + step * factor, prev.maxZoomIn),
                cursorX: prev.initialCoordinates.x,
                cursorY: prev.initialCoordinates.y,
                initialCoordinates: prev.initialCoordinates,
            })(prev);

            return { ...newState, hasAnimation };
        });
    };

    return (
        <Zoom.Provider value={zoom}>
            <SetZoom.Provider
                value={{
                    setZoom,
                    fitToScreen,
                    onZoomChange,
                }}
            >
                {children}
            </SetZoom.Provider>
        </Zoom.Provider>
    );
};
