// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Text } from '@geti/ui';
import { Add, Remove } from '@geti/ui/icons';
import { useSetZoom, useZoom } from 'src/components/zoom/zoom.store';

import { IconWrapper } from '../icon-wrapper.component';

export const ZoomSelector = () => {
    const { scale, maxZoomIn, initialCoordinates } = useZoom();
    const { onZoomChange } = useSetZoom();

    return (
        <>
            <ActionButton isQuiet aria-label='Zoom In' onPress={() => onZoomChange(1)} isDisabled={scale >= maxZoomIn}>
                <IconWrapper>
                    <Add />
                </IconWrapper>
            </ActionButton>

            <Flex justifyContent={'center'}>
                <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-25)' }}>
                    {(scale * 100).toFixed(0)}%
                </Text>
            </Flex>

            <ActionButton
                isQuiet
                aria-label='Zoom Out'
                onPress={() => onZoomChange(-1)}
                isDisabled={scale <= initialCoordinates.scale}
            >
                <IconWrapper>
                    <Remove />
                </IconWrapper>
            </ActionButton>
        </>
    );
};
