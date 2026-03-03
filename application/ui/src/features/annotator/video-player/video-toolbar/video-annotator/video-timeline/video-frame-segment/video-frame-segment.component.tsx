// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useNumberFormatter } from 'react-aria';

import { $api } from '../../../../../../../api/client';
import { Label } from '../../../../../../../constants/shared-types';
import { useVideoPlayer } from '../../../../video-player-provider.component';

import classes from './video-frame-segment.module.scss';

const getAriaLabel = (label: Label | undefined, isPrediction: boolean) => {
    if (isPrediction) {
        if (label === undefined) {
            return 'No prediction';
        }
        return `Predicted ${label.name}`;
    }

    if (label === undefined) {
        return 'No label';
    }
    return label.name;
};

type LabelSegmentProps = {
    label?: Label;
    striped: boolean;
    isLoading: boolean;
    isPrediction?: boolean;
};

const LabelSegment = ({ label, striped, isLoading, isPrediction = false }: LabelSegmentProps) => {
    if (isLoading) {
        return <View width='100%' height='size-100' UNSAFE_className={classes.loadingGradient} />;
    }

    const backgroundColor =
        label !== undefined ? label.color : striped ? 'var(--spectrum-global-color-gray-400)' : undefined;

    return (
        <div aria-label={getAriaLabel(label, isPrediction)} role='presentation'>
            <View
                width='100%'
                height='size-100'
                UNSAFE_className={`${classes.labelMarker} ${striped ? classes.stripedBackground : ''}`}
                UNSAFE_style={{ backgroundColor }}
            />
        </div>
    );
};

const SelectedFrameOverlay = () => {
    return (
        <View
            data-testid='selected'
            top={0}
            left={0}
            right={0}
            bottom={0}
            backgroundColor='static-white'
            position='absolute'
            UNSAFE_className={classes.activeFrameOverlay}
        />
    );
};

const CHUNK_SIZE = 30;

const useVideoFramesAnnotations = ({ frameNumber }: { frameNumber: number }) => {
    const projectId = useProjectIdentifier();
    const { videoFrame, step } = useVideoPlayer();

    const annotationChunkSize = CHUNK_SIZE * step;

    const frames = videoFrame.frame_count - 1;

    const startFrameIndex = Math.floor(frameNumber / annotationChunkSize) * annotationChunkSize;
    const endFrameIndex = Math.min(startFrameIndex + annotationChunkSize - 1, frames);

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/media/{media_id}/frames',
        {
            params: {
                path: {
                    project_id: projectId,
                    media_id: videoFrame.id,
                },
                query: {
                    frame_index_from: startFrameIndex,
                    frame_index_to: endFrameIndex,
                },
            },
        },
        {
            select: (data) => {
                return data.find(({ frame_index }) => frame_index === frameNumber);
            },
        }
    );
};

// TODO: Implement this properly.
// This hook should return the annotations and predictions for the current frame. Moreover we should fetch annotations
// and predictions for the previous and next frames as well (using start-end frame).
const useVideoTimelineQueries = ({ frameNumber }: { frameNumber: number }) => {
    const { data: videoFramesAnnotations, isPending: isVideoFramesAnnotationsPending } = useVideoFramesAnnotations({
        frameNumber,
    });

    const annotatedLabels =
        videoFramesAnnotations?.annotation_data?.annotations.flatMap((annotation) =>
            annotation.labels.map(({ id }) => id)
        ) ?? [];

    return {
        annotatedLabels,
        predictedLabels: [] as string[],
        isAnnotationLoading: false,
        isPredictionLoading: isVideoFramesAnnotationsPending,
    };
};

type VideoFrameSegmentProps = {
    isFirstFrame: boolean;
    isLastFrame: boolean;
    isSelectedFrame: boolean;
    labels: Label[];
    frameNumber: number;
    showTicks: boolean;
    colIndex: number;
    onClick: (frameNumber: number) => void;
};

export const VideoFrameSegment = ({
    isFirstFrame,
    isLastFrame,
    isSelectedFrame,
    showTicks,
    frameNumber,
    labels,
    colIndex,
    onClick,
}: VideoFrameSegmentProps) => {
    const formatter = useNumberFormatter({
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
        maximumSignificantDigits: 4,
        notation: 'compact',
    });

    const tickTextStyle = {
        left: isFirstFrame ? '50%' : 'initial',
        right: isLastFrame ? '0%' : 'initial',
    };

    const { annotatedLabels, predictedLabels, isAnnotationLoading, isPredictionLoading } = useVideoTimelineQueries({
        frameNumber,
    });

    return (
        <div
            onClick={() => {
                onClick(frameNumber);
            }}
            className={classes.videoFrameSegmentContainer}
        >
            <View marginTop={'-0.5rem'} paddingBottom={'0.5rem'} position={'relative'}>
                {showTicks ? (
                    <div className={classes.ticksText} style={tickTextStyle}>
                        {formatter.format(frameNumber)}f
                        <div className={classes.ticksIndicator} />
                    </div>
                ) : (
                    <></>
                )}
            </View>

            {/* TODO: Check if we need this overlay */}
            {/*{isActiveFrame || isFilteredFrame ? <ActiveFrameOverlay /> : <></>}*/}

            {isSelectedFrame && <SelectedFrameOverlay />}

            <Flex gap='size-100' direction='column' marginTop='size-100'>
                {labels.map((label, rowIndex) => {
                    return (
                        <div
                            role='gridcell'
                            key={label.id}
                            aria-colindex={colIndex + 1}
                            aria-rowindex={rowIndex + 1}
                            aria-label={`Label ${label.name} in frame number ${frameNumber}`}
                        >
                            <Flex gap='size-10' height='size-225' direction='column' marginEnd={'size-10'}>
                                <LabelSegment
                                    striped={false}
                                    isLoading={isAnnotationLoading}
                                    label={annotatedLabels.includes(label.id) ? label : undefined}
                                />
                                <LabelSegment
                                    isPrediction
                                    striped={true}
                                    isLoading={isPredictionLoading}
                                    label={predictedLabels.includes(label.id) ? label : undefined}
                                />
                            </Flex>
                        </div>
                    );
                })}
            </Flex>
        </div>
    );
};
