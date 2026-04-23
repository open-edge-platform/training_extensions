// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Item, Key, Picker } from '@geti/ui';

import { $api } from '../../../api/client';
import { DeviceInfo } from '../../../constants/shared-types';
import { usePatchPipeline, usePipeline } from '../../../hooks/api/pipeline.hook';
import { useProjectIdentifier } from '../../../hooks/use-project-identifier.hook';

// Generate device id based on type and index (if available) to ensure uniqueness
// in case of multiple devices of the same type (e.g., multiple GPUs)
const getDeviceId = (device: DeviceInfo): string =>
    device.index != null ? `${device.type}-${device.index}` : device.type;

export const InferenceDevices = () => {
    const { data: devices } = $api.useSuspenseQuery('get', '/api/system/devices/inference');
    const { data: pipeline } = usePipeline();
    const projectId = useProjectIdentifier();
    const [selectedKey, setSelectedKey] = useState<Key | null>(pipeline.device);
    const updatePipeline = usePatchPipeline();

    const items = devices.map((device) => ({ ...device, id: getDeviceId(device) }));

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        setSelectedKey(key);
        updatePipeline.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { device: key },
            },
            {
                onError: (error) => {
                    if (error) {
                        setSelectedKey(pipeline.device);
                    }
                },
            }
        );
    };

    return (
        <Picker
            maxWidth='size-3000'
            aria-label='inference compute'
            labelAlign='end'
            labelPosition='side'
            items={items}
            onSelectionChange={handleChange}
            selectedKey={selectedKey}
        >
            {(device) => <Item key={device.id}>{device.name}</Item>}
        </Picker>
    );
};
