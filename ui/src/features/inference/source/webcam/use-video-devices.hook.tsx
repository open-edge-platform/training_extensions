// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { getBrowserPermissions, getVideoDevices, UserCameraPermission } from './util';

export const useVideoDevices = () => {
    const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
    const [userPermissions, setUserPermissions] = useState(UserCameraPermission.PENDING);
    const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);

    useEffect(() => {
        getBrowserPermissions().then(({ permissions, stream }) => {
            // Stop the stream because react-webcam starts its own stream
            stream && stream.getTracks().forEach((track) => track.stop());

            getVideoDevices().then((newDevices) => {
                setVideoDevices(newDevices);
                setUserPermissions(permissions);
                setSelectedDeviceId(newDevices.at(0)?.deviceId ?? '');
            });
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return { videoDevices, userPermissions, selectedDeviceId, setSelectedDeviceId };
};
