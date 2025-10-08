// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton } from '@geti/ui';
import { FitScreen } from '@geti/ui/icons';

import { useSetZoom } from '../../zoom/zoom.provider';
import { IconWrapper } from '../icon-wrapper.component';

export const ZoomFitScreen = () => {
    const setZoom = useSetZoom();

    const handleFitScreen = () => {
        setZoom((prev) => ({
            ...prev,
            scale: prev.initialCoordinates.scale,
            translate: { x: prev.initialCoordinates.x, y: prev.initialCoordinates.y },
        }));
    };

    return (
        <ActionButton isQuiet onPress={handleFitScreen} aria-label='Fit to screen'>
            <IconWrapper>
                <FitScreen />
            </IconWrapper>
        </ActionButton>
    );
};
