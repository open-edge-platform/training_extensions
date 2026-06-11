// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// https://developer.mozilla.org/en-US/docs/Web/API/MediaError/code#media_error_code_constants
export enum VideoPlayerErrorReason {
    // The fetching of the associated resource was aborted by the user's request.
    MEDIA_ERR_ABORTED = 1,
    // Some kind of network error occurred which prevented the media from being successfully
    // fetched, despite having previously been available.
    MEDIA_ERR_NETWORK = 2,
    // Despite having previously been determined to be usable, an error occurred while trying
    // to decode the media resource, resulting in an error.
    MEDIA_ERR_DECODE = 3,
    MEDIA_ERR_SRC_NOT_SUPPORTED = 4,
}

export const getVideoErrorMessage = (error: VideoPlayerErrorReason | null) => {
    if (error === null) {
        return null;
    }

    if ([VideoPlayerErrorReason.MEDIA_ERR_ABORTED, VideoPlayerErrorReason.MEDIA_ERR_NETWORK].includes(error)) {
        return 'Cannot play video due to a network error, please refresh and try again.';
    }

    if ([VideoPlayerErrorReason.MEDIA_ERR_DECODE, VideoPlayerErrorReason.MEDIA_ERR_SRC_NOT_SUPPORTED].includes(error)) {
        return 'Unable to play video, please try a different browser or a different video format (mp4/webm).';
    }
};
