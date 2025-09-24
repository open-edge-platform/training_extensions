// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Loading, Picker } from '@geti/ui';

import { PermissionError } from './permissions-error.component';
import { useVideoDevices } from './use-video-devices.hook';
import { hasPermissionsDenied, isPermissionPending } from './util';

export const Webcam = () => {
    const { videoDevices, userPermissions, selectedDeviceId, setSelectedDeviceId } = useVideoDevices();

    if (isPermissionPending(userPermissions)) {
        return <Loading aria-label='permissions pending' size='S' />;
    }

    if (hasPermissionsDenied(userPermissions)) {
        return <PermissionError />;
    }

    return (
        <Picker
            width={'100%'}
            label={'Device'}
            items={videoDevices}
            aria-label={'devices'}
            selectedKey={selectedDeviceId}
            placeholder={'Integrated Cameras'}
            onSelectionChange={(key) => setSelectedDeviceId(String(key))}
        >
            {({ deviceId, label }) => <Item key={deviceId}>{label}</Item>}
        </Picker>
    );
};
