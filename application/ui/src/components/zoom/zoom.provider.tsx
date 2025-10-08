// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

export type ZoomState = {
    scale: number;
    maxZoomIn: number;
    translate: { x: number; y: number };
    initialCoordinates: { scale: number; x: number; y: number };
};

export const Zoom = createContext<ZoomState>({
    scale: 1.0,
    maxZoomIn: 1,
    translate: { x: 0, y: 0 },
    initialCoordinates: { scale: 1.0, x: 0, y: 0 },
});
const SetZoom = createContext<Dispatch<SetStateAction<ZoomState>> | null>(null);

export const useZoom = () => {
    return useContext(Zoom);
};

export const useSetZoom = () => {
    const context = useContext(SetZoom);

    if (!context) {
        throw new Error('');
    }

    return context;
};

export const ZoomProvider = ({ children }: { children: ReactNode }) => {
    // 1. Add translate restrictions - min max
    const [zoom, setZoom] = useState<ZoomState>({
        scale: 1.0,
        maxZoomIn: 1,
        translate: { x: 0, y: 0 },
        initialCoordinates: { scale: 1.0, x: 0, y: 0 },
    });

    return (
        <Zoom.Provider value={zoom}>
            <SetZoom.Provider value={setZoom}>{children}</SetZoom.Provider>
        </Zoom.Provider>
    );
};
