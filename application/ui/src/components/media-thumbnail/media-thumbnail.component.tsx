// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import type { Media, MediaVideo } from '../../constants/shared-types';
import { isVideo } from '../../shared/media-item-utils';
import { formatCompactDuration } from './util';

import classes from './media-thumbnail.module.scss';

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
    item: Pick<Media, 'type'> | Pick<MediaVideo, 'type' | 'frame_count' | 'annotated_frame_count' | 'duration'>;
};

type VideoIndicatorProps = {
    duration: number;
};

const VideoIndicator = ({ duration }: VideoIndicatorProps) => {
    return (
        <Flex
            gap={'size-50'}
            left={'size-50'}
            bottom={'size-50'}
            position={'absolute'}
            alignItems={'center'}
            UNSAFE_className={classes.videoIndicator}
        >
            {formatCompactDuration(duration)}
        </Flex>
    );
};

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt, item }: MediaThumbnailProps) => {
    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick} className={classes.imgContainer}>
            <img src={url} alt={alt} className={classes.img} />
            {isVideo(item) && <VideoIndicator duration={item.duration} />}
        </div>
    );
};
