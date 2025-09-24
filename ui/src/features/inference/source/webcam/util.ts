// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uniqBy } from 'lodash-es';

export enum UserCameraPermission {
    GRANTED = 'granted',
    DENIED = 'denied',
    ERRORED = 'errored',
    PENDING = 'pending',
}

// For now we only care about these two. For the complete list please check
// https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
export enum UserCameraPermissionError {
    NOT_ALLOWED = 'NotAllowedError',
    NOT_FOUND = 'NotFoundError',
}

export const GETI_CAMERA_INDEXEDDB_INSTANCE_NAME = 'geti-camera';

export const getVideoUserMedia = () => navigator.mediaDevices.getUserMedia({ video: true });

export const isVideoInput = (mediaDevice: MediaDeviceInfo) => mediaDevice.kind === 'videoinput';

export const getVideoDevices = async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();

    const videoDevices = devices.filter(isVideoInput);

    return uniqBy(videoDevices, (device) => device.deviceId);
};

export const getPermissionError = (error: unknown) => {
    const errorType = error instanceof Error ? error.message : '';

    return errorType === UserCameraPermissionError.NOT_ALLOWED
        ? UserCameraPermission.DENIED
        : UserCameraPermission.ERRORED;
};

export const getBrowserPermissions = async () => {
    try {
        const stream = await getVideoUserMedia();
        return { permissions: UserCameraPermission.GRANTED, stream };
    } catch (error: unknown) {
        return { permissions: getPermissionError(error), stream: null };
    }
};

export const hasPermissionsDenied = (permissions: UserCameraPermission) =>
    [UserCameraPermission.ERRORED, UserCameraPermission.DENIED].includes(permissions);

export const isPermissionPending = (permissions: UserCameraPermission) => {
    return permissions === UserCameraPermission.PENDING;
};
