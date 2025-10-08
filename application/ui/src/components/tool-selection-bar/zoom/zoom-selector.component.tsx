// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { clampBetween } from '@geti/smart-tools/utils';
import { ActionButton, Flex } from '@geti/ui';
import { Add, Remove } from '@geti/ui/icons';

import { getZoomState } from '../../zoom/util';
import { useSetZoom, useZoom } from '../../zoom/zoom.provider';
import { IconWrapper } from '../icon-wrapper.component';

const ZOOM_STEP_DIVISOR = 10;

export const ZoomSelector = () => {
    const zoom = useZoom();
    const setZoom = useSetZoom();

    const handleZoomChange = (factor: number) => {
        const step = (zoom.maxZoomIn - zoom.initialCoordinates.scale) / ZOOM_STEP_DIVISOR;

        setZoom(
            getZoomState({
                newScale: clampBetween(zoom.initialCoordinates.scale, zoom.scale + step * factor, zoom.maxZoomIn),
                cursorX: zoom.initialCoordinates.x,
                cursorY: zoom.initialCoordinates.y,
                initialCoordinates: zoom.initialCoordinates,
            })
        );
    };

    return (
        <>
            <ActionButton
                isQuiet
                aria-label='Zoom In'
                onPress={() => handleZoomChange(1)}
                isDisabled={zoom.scale >= zoom.maxZoomIn}
            >
                <IconWrapper>
                    <Add />
                </IconWrapper>
            </ActionButton>

            <Flex width={'size-600'} justifyContent={'center'}>
                {(zoom.scale * 100).toFixed(1)}%
            </Flex>

            <ActionButton
                isQuiet
                aria-label='Zoom Out'
                onPress={() => handleZoomChange(-1)}
                isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
            >
                <IconWrapper>
                    <Remove />
                </IconWrapper>
            </ActionButton>
        </>
    );
};
