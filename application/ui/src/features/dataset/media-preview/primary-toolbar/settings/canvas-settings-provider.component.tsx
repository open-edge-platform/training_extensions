// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, use, useState } from 'react';

interface CanvasSettingsProviderProps {
    children: ReactNode;
}

export interface CanvasSettingsState {
    annotationFillOpacity: { value: number; defaultValue: number };
    annotationBorderOpacity: { value: number; defaultValue: number };
    imageBrightness: { value: number; defaultValue: number };
    imageContrast: { value: number; defaultValue: number };
    imageSaturation: { value: number; defaultValue: number };
    pixelView: { value: boolean; defaultValue: boolean };
    hideLabels: { value: boolean; defaultValue: boolean };
}

interface CanvasSettingsContextProps {
    canvasSettings: CanvasSettingsState;
    setCanvasSettings: Dispatch<SetStateAction<CanvasSettingsState>>;
}

const CanvasSettingsContext = createContext<CanvasSettingsContextProps | null>(null);

export const DEFAULT_CANVAS_SETTINGS: CanvasSettingsState = {
    annotationFillOpacity: {
        value: 0.5,
        defaultValue: 0.5,
    },
    annotationBorderOpacity: {
        value: 1,
        defaultValue: 1,
    },
    imageSaturation: {
        value: 0,
        defaultValue: 0,
    },
    imageBrightness: {
        value: 0,
        defaultValue: 0,
    },
    imageContrast: {
        value: 0,
        defaultValue: 0,
    },
    pixelView: {
        value: false,
        defaultValue: false,
    },
    hideLabels: {
        value: false,
        defaultValue: false,
    },
};

export const CanvasSettingsProvider = ({ children }: CanvasSettingsProviderProps) => {
    const [canvasSettings, setCanvasSettings] = useState<CanvasSettingsState>(DEFAULT_CANVAS_SETTINGS);

    return <CanvasSettingsContext value={{ canvasSettings, setCanvasSettings }}>{children}</CanvasSettingsContext>;
};

export const useCanvasSettings = () => {
    const context = use(CanvasSettingsContext);

    if (context === null) {
        throw new Error('useCanvasSettings must be used within a CanvasSettingsProvider');
    }

    return context;
};
