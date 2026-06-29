// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, useNumberFormatter, View } from '@geti-ui/ui';

const useFormatFrames = (frames: number) => {
    const formatter = useNumberFormatter({
        minimumFractionDigits: 0,
        maximumFractionDigits: 1,
        notation: 'compact',
    });

    return formatter.format(frames);
};

interface FrameNumberIndicatorProps {
    frameNumber: number;
}

export const FrameNumberIndicator = ({ frameNumber }: FrameNumberIndicatorProps) => {
    const formattedFrames = useFormatFrames(frameNumber);

    return (
        <View
            position='absolute'
            backgroundColor='gray-50'
            right={4}
            top={4}
            borderRadius={'regular'}
            paddingX='size-75'
            paddingY='size-25'
            UNSAFE_style={{ color: 'white', fontSize: dimensionValue('size-150') }}
        >
            {formattedFrames}f
        </View>
    );
};
