// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Text } from '@geti/ui';
import { Add, Remove } from '@geti/ui/icons';

import { IconWrapper } from '../../../../../components/icon-wrapper/icon-wrapper.component';
import { useSetZoom, useZoom } from '../../../../../components/zoom/zoom.provider';

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

            <Flex justifyContent={'center'}>
                <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-25)' }}>
                    {(zoom.scale * 100).toFixed(0)}%
                </Text>
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
