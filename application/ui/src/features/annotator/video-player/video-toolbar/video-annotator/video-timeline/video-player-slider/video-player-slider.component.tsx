// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, RefObject, useEffect, useRef, useState } from 'react';

import { useDebouncedCallback } from 'hooks/use-debounced-callback/use-debounced-callback.hook';
import { defer } from 'lodash-es';
import { useHover } from 'react-aria';

import { Media } from '../../../../../../../constants/shared-types';
import { ThumbnailPreview } from './thumbnail-preview.component';
import { VideoSlider } from './video-slider.component';

import classes from './video-slider.module.scss';

type VideoPlayerSliderProps = {
    mediaItem: Media;
    step: number;
    frameNumber: number;
    sizePerSquare: number;
    frameOffset?: number;
    ref?: RefObject<HTMLDivElement | null>;
};

const THUMBNAIL_DELAY = 1000;

const useShowThumbnail = () => {
    const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    const [thumbnailVideoFrame, setThumbnailVideoFrame] = useState<null | number>(null);
    const [showThumbnail, setShowThumbnail] = useState(false);
    const [thumbnailPosition, setThumbnailPosition] = useState<null | number>(null);
    const setThumbnailVideoFrameDebounced = useDebouncedCallback(setThumbnailVideoFrame, 200);

    const { hoverProps } = useHover({
        onHoverStart: () => {
            timeoutRef.current = setTimeout(() => setShowThumbnail(true), THUMBNAIL_DELAY);
        },
        onHoverEnd: () => {
            timeoutRef.current && clearTimeout(timeoutRef.current);
            setShowThumbnail(false);
            setThumbnailPosition(null);
            setThumbnailVideoFrameDebounced(null);
        },
    });

    useEffect(() => {
        return () => {
            timeoutRef.current && clearTimeout(timeoutRef.current);
        };
    }, []);

    return {
        hoverProps,
        onShowThumbnail: setShowThumbnail,
        showThumbnail,
        thumbnailVideoFrame,
        thumbnailPosition,
        onSetThumbnailPosition: setThumbnailPosition,
        onSetThumbnailVideoFrameDebounced: setThumbnailVideoFrameDebounced,
    };
};

const getFrameNumber = (x: number, width: number, minValue: number, maxValue: number, step: number) => {
    const frameNumber = Math.min(
        Math.floor(maxValue / step) * step,
        Math.max(0, Math.round(Math.round((x / width) * (maxValue - minValue)) / step) * step)
    );

    return frameNumber;
};

const getContainerScroll = (ref?: RefObject<HTMLDivElement | null>) => {
    if (ref === undefined) {
        return 0;
    }

    if (ref.current) {
        return ref.current.scrollLeft;
    }

    return 0;
};

export const blurActiveInput = (isFocused: boolean): void => {
    const element = document.activeElement;

    if (isFocused && element?.nodeName === 'INPUT') {
        defer(() => (element as HTMLInputElement).blur());
    }
};

export const VideoPlayerSlider = ({
    ref,
    mediaItem,
    step,
    frameNumber,
    sizePerSquare,
    frameOffset = 0,
}: VideoPlayerSliderProps) => {
    const [sliderValue, setSliderValue] = useState<number>(frameNumber);

    const {
        hoverProps,
        showThumbnail,
        onShowThumbnail,
        onSetThumbnailPosition,
        onSetThumbnailVideoFrameDebounced,
        thumbnailVideoFrame,
        thumbnailPosition,
    } = useShowThumbnail();

    const framesCount = Number(mediaItem.frame_count);
    const minValue = 0;
    const maxValue = framesCount - 1;
    const lastFrame = framesCount - step;
    const isLastFrame = sliderValue >= lastFrame;
    const containerScrollLeft = getContainerScroll(ref);

    const highlightedFrames: number[] = [];
    const buffers = undefined;

    const handlePointerMove = (event: PointerEvent<HTMLDivElement>): void => {
        const rect = event.currentTarget.getBoundingClientRect();

        const thumbnailPosX = Math.max(0, event.clientX - rect.x) - containerScrollLeft;

        const thumbnailFrameNumber = getFrameNumber(
            thumbnailPosX - frameOffset + containerScrollLeft,
            rect.width,
            minValue,
            maxValue,
            step
        );

        onSetThumbnailPosition(thumbnailPosX);
        onSetThumbnailVideoFrameDebounced(thumbnailFrameNumber);
    };

    return (
        <div className={classes.sliderWrapper} onPointerMove={handlePointerMove} {...hoverProps}>
            <VideoSlider
                isFilled
                width={'100%'}
                aria-label={'Video timeline'}
                showValueLabel={false}
                minValue={minValue}
                maxValue={maxValue}
                defaultValue={sliderValue}
                value={sliderValue}
                onChange={(newFrameNumber) => {
                    setSliderValue(newFrameNumber);
                    onShowThumbnail(true);
                }}
                onChangeEnd={(_newFrameNumber) => {
                    // selectFrame(newFrameNumber);
                    blurActiveInput(true);
                }}
                step={step}
                buffers={buffers}
                highlightedFrames={highlightedFrames}
                isLastFrame={isLastFrame}
                sizePerSquare={sizePerSquare}
                leftOffset={frameOffset}
            />
            {showThumbnail && thumbnailVideoFrame !== null && thumbnailPosition !== null && (
                <ThumbnailPreview
                    x={thumbnailPosition}
                    width={100}
                    height={100}
                    videoFrame={thumbnailVideoFrame}
                    mediaItem={mediaItem}
                />
            )}
        </div>
    );
};
