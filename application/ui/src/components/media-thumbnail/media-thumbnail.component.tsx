// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, ContextualHelp, Flex, Text } from '@geti/ui';

import type { Media, MediaVideo } from '../../constants/shared-types';
import { isVideo } from '../../shared/media-item-utils';

import classes from './media-thumbnail.module.scss';

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
    item: Pick<Media, 'type'> | Pick<MediaVideo, 'type' | 'frame_count' | 'annotated_frame_count'>;
};

type VideoIndicatorProps = {
    frameCount: number;
    annotatedFrameCount: number;
};

const VideoIndicator = ({ frameCount, annotatedFrameCount }: VideoIndicatorProps) => {
    return (
        <Flex
            gap={'size-50'}
            left={'size-50'}
            bottom={'size-50'}
            position={'absolute'}
            alignItems={'center'}
            UNSAFE_className={classes.videoIndicator}
        >
            {`${annotatedFrameCount} / ${frameCount} ${frameCount !== 1 ? 'frames' : 'frame'}`}

            <ContextualHelp
                variant='info'
                UNSAFE_className={classes.videoIndicatorDetails}
                aria-label='annotated frames'
            >
                <Content>
                    <Text>Total frames: {frameCount}</Text>
                    <br />
                    <Text>Number of annotated frames: {annotatedFrameCount}</Text>
                </Content>
            </ContextualHelp>
        </Flex>
    );
};

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt, item }: MediaThumbnailProps) => {
    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick} className={classes.imgContainer}>
            <img src={url} alt={alt} className={classes.img} />
            {isVideo(item) && (
                <VideoIndicator frameCount={item.frame_count} annotatedFrameCount={item.annotated_frame_count} />
            )}
        </div>
    );
};
