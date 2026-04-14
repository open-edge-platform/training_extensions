// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const CHUNK_SIZE = 30;

export const getVideoFrameRangeIndexes = ({
    frameSkip,
    frames,
    frameNumber,
    chunkSize = CHUNK_SIZE,
}: {
    frameNumber: number;
    frames: number;
    frameSkip: number;
    chunkSize?: number;
}) => {
    const annotationChunkSize = chunkSize * frameSkip;

    const startFrameIndex = Math.min(Math.floor(frameNumber / annotationChunkSize) * annotationChunkSize, frames);
    const endFrameIndex = Math.min(startFrameIndex + annotationChunkSize - 1, frames);

    return { startFrameIndex, endFrameIndex };
};
