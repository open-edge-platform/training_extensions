// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Media } from '../../../../../../../constants/shared-types';
import { VideoSlider } from './video-slider.component';

import classes from './video-slider.module.scss';

type VideoPlayerSliderProps = {
    media: Media;
    step: number;
    frameNumber: number;
    sizePerSquare: number;
    frameOffset?: number;
};

export const VideoPlayerSlider = ({ media, step, frameNumber, sizePerSquare, frameOffset }: VideoPlayerSliderProps) => {
    const [sliderValue, setSliderValue] = useState<number>(frameNumber);
    const minValue = 0;
    const framesCount = Number(media.frame_count);
    const lastFrame = framesCount - step;
    const isLastFrame = sliderValue >= lastFrame;

    const highlightedFrames: number[] = [];
    const buffers = undefined;

    return (
        <div className={classes.sliderWrapper}>
            <VideoSlider
                isFilled
                width={'100%'}
                aria-label={'Video timeline'}
                showValueLabel={false}
                minValue={minValue}
                maxValue={framesCount}
                defaultValue={sliderValue}
                value={sliderValue}
                onChange={setSliderValue}
                step={step}
                buffers={buffers}
                highlightedFrames={highlightedFrames}
                isLastFrame={isLastFrame}
                sizePerSquare={sizePerSquare}
                leftOffset={frameOffset}
            />
        </div>
    );
};
