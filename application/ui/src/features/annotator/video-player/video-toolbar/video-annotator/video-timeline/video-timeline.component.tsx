// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import { useSizeHook } from 'hooks/use-size.hook';
import useVirtual from 'react-cool-virtual';

import { type Label } from '../../../../../../constants/shared-types';
import { AnnotatorMode } from '../../../../../dataset/media-preview/secondary-toolbar/annotator-modes/mode';
import { useVideoPlayer } from '../../../video-player-provider.component';
import { VideoFrameSegments } from './video-frame-segment/video-frame-segments.component';
import { VideoPlayerSlider } from './video-player-slider/video-player-slider.component';

import classes from './video-timeline.module.scss';

type VideoTimelineProps = {
    labels: Label[];
    mode: AnnotatorMode;
};

const MIN_SIZE_OF_SEGMENT = 2 * 8;

export const VideoTimeline = ({ labels, mode }: VideoTimelineProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const size = useSizeHook(containerRef);
    const { videoFrame, videoControls, step } = useVideoPlayer();
    const { isPlaying } = videoControls;

    const frameNumber = videoFrame.frame_number;
    const totalFrames = videoFrame.frame_count;

    const totalSegments = Math.max(Math.ceil(totalFrames / step), 0);
    const sizePerSquare = size === undefined ? 0 : Math.max(MIN_SIZE_OF_SEGMENT, size.width / totalSegments);
    const frameOffset = Math.round(sizePerSquare / 2);

    const { outerRef, innerRef, items, scrollToItem } = useVirtual<HTMLDivElement, HTMLDivElement>({
        horizontal: true,
        itemCount: totalSegments,
        itemSize: sizePerSquare,
        overscanCount: 20,
    });

    useEffect(() => {
        const segmentIndex = Math.round(frameNumber / step);

        scrollToItem({ index: segmentIndex, align: 'center' });
    }, [scrollToItem, frameNumber, step]);

    return (
        <div ref={containerRef}>
            <div ref={outerRef} style={{ overflow: 'auto', width: size?.width }}>
                <div style={{ width: sizePerSquare * totalSegments }} className={classes.timelineSliderWrapper}>
                    <VideoPlayerSlider
                        ref={outerRef}
                        videoFrame={videoFrame}
                        step={step}
                        frameNumber={frameNumber}
                        sizePerSquare={sizePerSquare}
                        frameOffset={frameOffset}
                        selectFrame={videoControls.goto}
                    />
                </div>
                <VideoFrameSegments
                    mode={mode}
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
