// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export enum AlgorithmType {
    // For Segment Anything we use two workers so that the encoder and decoder are run on separate threads
    SEGMENT_ANYTHING_ENCODER = 'SEGMENT_ANYTHING_ENCODER',
    SEGMENT_ANYTHING_DECODER = 'SEGMENT_ANYTHING_DECODER',
}
