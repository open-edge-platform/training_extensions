// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Image, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { Media } from '../../../../../../../constants/shared-types';
import { getThumbnailUrl } from '../../../../../../../shared/media-url.utils';
import { formatDurationText } from '../../../time-utils';
import { FrameNumberIndicator } from './frame-number-indicator.component';

interface ThumbnailPreviewProps {
    videoFrame: number;
    mediaItem: Media;
    width: number;
    height: number;
    x: number;
}

export const ThumbnailPreview = ({ mediaItem, videoFrame: frameNumber, width, height, x }: ThumbnailPreviewProps) => {
    // const constructVideoFrame = useConstructVideoFrame(mediaItem);
    // const videoFrame = constructVideoFrame(frameNumber) as VideoFrame;
    const projectIdentifier = useProjectIdentifier();

    const fps = Number(mediaItem.fps);
    // TODO: Replace it with the video frame thumbnail when the endpoint will be ready.
    const src = getThumbnailUrl(projectIdentifier, '7301f742-fd9e-41af-8eeb-77bd9e771eec');

    const durationText = formatDurationText(frameNumber / fps);

    return (
        <View
            position='absolute'
            top={-height - 25}
            left={x - width / 2}
            overflow='hidden'
            borderRadius='small'
            UNSAFE_style={{ boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.38)', pointerEvents: 'none' }}
        >
            <View position='fixed'>
                <FrameNumberIndicator frameNumber={frameNumber} />
                <Image
                    src={src}
                    alt={`Thumbnail for frame ${frameNumber}`}
                    objectFit='cover'
                    height={height}
                    width={width}
                />
                <View
                    backgroundColor='gray-50'
                    paddingX='size-75'
                    paddingY='size-25'
                    alignSelf='center'
                    UNSAFE_style={{ color: 'white', fontSize: '11px', textAlign: 'center' }}
                >
                    {durationText}
                </View>
                <View backgroundColor='static-white' width='1px' marginX='auto' height='size-200'>
                    &nbsp;
                </View>
            </View>
        </View>
    );
};
