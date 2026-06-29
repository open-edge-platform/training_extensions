// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { DeviceInfo } from '../../constants/shared-types';

// Generate device id based on type and index (if available) to ensure uniqueness
// in case of multiple devices of the same type (e.g., multiple GPUs)
const getDeviceId = (device: DeviceInfo): string =>
    device.index != null ? `${device.type}-${device.index}` : device.type;

type InferenceDevicesProps = {
    selectedKey: string;
    onSelectionChange: (selectedKey: string) => void;
    isQuiet?: boolean;
    isDisabled?: boolean;
    ariaLabel?: string;
};

export const InferenceDevices = ({
    selectedKey,
    onSelectionChange,
    isDisabled = false,
    isQuiet = false,
    ariaLabel,
}: InferenceDevicesProps) => {
    const { data: devices } = $api.useSuspenseQuery('get', '/api/system/devices/inference');

    const items = devices.map((device) => ({ ...device, id: getDeviceId(device) }));

    const handleSelectionChange = (key: Key | null): void => {
        if (key === null || key === selectedKey) {
            return;
        }

        onSelectionChange(key.toString());
    };

    return (
        <Picker
            isQuiet={isQuiet}
            maxWidth='size-3000'
            items={items}
            onSelectionChange={handleSelectionChange}
            selectedKey={selectedKey}
            isDisabled={isDisabled}
            aria-label={ariaLabel}
        >
            {(device) => <Item key={device.id}>{device.name}</Item>}
        </Picker>
    );
};
