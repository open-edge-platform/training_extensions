// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const CHUNK_SIZE = 30;

export const getVideoFrameRangeIndexes = ({
    frameSkip,
    frames,
    frameNumber,
}: {
    frameNumber: number;
    frames: number;
    frameSkip: number;
}) => {
    const annotationChunkSize = CHUNK_SIZE * frameSkip;

    const startFrameIndex = Math.floor(frameNumber / annotationChunkSize) * annotationChunkSize;
    const endFrameIndex = Math.min(startFrameIndex + annotationChunkSize - 1, frames);

    return { startFrameIndex, endFrameIndex };
};
