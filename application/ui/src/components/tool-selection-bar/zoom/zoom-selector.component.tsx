// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Add, Remove } from '@geti/ui/icons';

import { useSetZoom, useZoom } from '../../zoom/zoom.provider';
import { IconWrapper } from '../icon-wrapper.component';

export const ZoomSelector = () => {
    const zoom = useZoom();
    const { onZoomChange } = useSetZoom();

    return (
        <>
            <ActionButton
                isQuiet
                aria-label='Zoom In'
                onPress={() => onZoomChange(1)}
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
                onPress={() => onZoomChange(-1)}
                isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
            >
                <IconWrapper>
                    <Remove />
                </IconWrapper>
            </ActionButton>
        </>
    );
};
