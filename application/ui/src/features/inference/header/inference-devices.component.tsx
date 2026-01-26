// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Item, Key, Picker, toast } from '@geti/ui';

import { $api } from '../../../api/client';
import { usePatchPipeline, usePipeline } from '../../../hooks/api/pipeline.hook';
import { useProjectIdentifier } from '../../../hooks/use-project-identifier.hook';

export const InferenceDevices = () => {
    const { data: devices } = $api.useSuspenseQuery('get', '/api/system/devices/inference');
    const { data: pipeline } = usePipeline();
    const projectId = useProjectIdentifier();
    const [selectedKey, setSelectedKey] = useState<Key | null>(pipeline.device);
    const updatePipeline = usePatchPipeline();

    const options = devices.map((device) => ({ id: device.type, name: device.name }));

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
                        toast({ type: 'error', message: String(error.detail) });
                        setSelectedKey(pipeline.device);
                    }
                },
            }
        );
    };

    return (
        <Picker
            maxWidth='size-3000'
            label='Inference Compute: '
            aria-label='inference compute'
            labelAlign='end'
            labelPosition='side'
            items={options}
            onSelectionChange={handleChange}
            selectedKey={selectedKey}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
