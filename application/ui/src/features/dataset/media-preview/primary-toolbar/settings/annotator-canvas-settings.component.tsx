// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, ReactNode } from 'react';

import { useCanvasSettings } from './canvas-settings-provider.component';

interface AnnotatorCanvasSettingsProps {
    children: ReactNode;
}

const offsetToPercentage = (value: number) => {
    return `${value + 100}%`;
};

export const AnnotatorCanvasSettings = ({ children }: AnnotatorCanvasSettingsProps) => {
    const { canvasSettings } = useCanvasSettings();

    return (
        <div
            style={
                {
                    height: '100%',
                    '--annotation-fill-opacity': canvasSettings.annotationFillOpacity.value,
                    '--annotation-border-opacity': canvasSettings.annotationBorderOpacity.value,
                    '--image-brightness': offsetToPercentage(canvasSettings.imageBrightness.value),
                    '--image-saturation': offsetToPercentage(canvasSettings.imageSaturation.value),
                    '--image-contrast': offsetToPercentage(canvasSettings.imageContrast.value),
                    '--pixel-view': canvasSettings.pixelView.value ? 'pixelated' : 'auto',
                } as CSSProperties
            }
        >
            {children}
        </div>
    );
};
