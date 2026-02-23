// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import { useSizeHook } from 'hooks/use-size.hook';
import useVirtual from 'react-cool-virtual';

import { type Label } from '../../../../../../constants/shared-types';
import { useVideoPlayer } from '../../../video-player-provider.component';
import { VideoFrameSegments } from './video-frame-segment/video-frame-segments.component';
import { VideoPlayerSlider } from './video-player-slider/video-player-slider.component';

import classes from './video-timeline.module.scss';

type VideoTimelineProps = {
    labels: Label[];
};

const MIN_SIZE_OF_SEGMENT = 2 * 8;

export const VideoTimeline = ({ labels }: VideoTimelineProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const size = useSizeHook(containerRef);
    const { videoFrame, videoControls } = useVideoPlayer();
    const { isPlaying } = videoControls;

    const frameNumber = videoFrame.frame_number;
    const step = videoFrame.frame_stride;
    const totalFrames = videoFrame.frame_count;

    const totalSegments = Math.ceil(totalFrames / step);
    const sizePerSquare = size === undefined ? 0 : Math.max(MIN_SIZE_OF_SEGMENT, size.width / totalSegments);
    const frameOffset = Math.round(sizePerSquare / 2);

    const { outerRef, innerRef, items, scrollToItem } = useVirtual<HTMLDivElement, HTMLDivElement>({
        horizontal: true,
        itemCount: totalSegments,
        itemSize: sizePerSquare,
        overscanCount: 5,
    });

    useEffect(() => {
        scrollToItem({ index: Math.round(frameNumber / step), align: 'center', smooth: true });
    }, [scrollToItem, frameNumber, step]);

    return (
        <div ref={containerRef}>
            <div ref={outerRef} style={{ overflow: 'auto', width: size?.width }}>
                <div style={{ width: sizePerSquare * totalSegments }} className={classes.timelineSliderWrapper}>
                    <VideoPlayerSlider
                        ref={outerRef}
                        mediaItem={videoFrame}
                        step={step}
                        frameNumber={frameNumber}
                        sizePerSquare={sizePerSquare}
                        frameOffset={frameOffset}
                        selectFrame={videoControls.goto}
                    />
                </div>
                <VideoFrameSegments
                    frameNumber={frameNumber}
                    ref={innerRef}
                    items={items}
                    containerSize={size?.width ?? 0}
                    labels={labels}
                    step={step}
                    totalSegments={totalSegments}
                    minSizeOfSegment={MIN_SIZE_OF_SEGMENT}
                    isPlaying={isPlaying}
                    selectFrame={videoControls.goto}
                />
            </div>
        </div>
    );
};
