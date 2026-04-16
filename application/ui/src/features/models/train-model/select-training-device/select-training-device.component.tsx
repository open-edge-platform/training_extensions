// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import type { TrainingDevice } from '../../../../constants/shared-types';
import { createTrainingDeviceKey, useTrainModelState } from '../train-model-provider.component';

const formatTrainingDeviceMemory = (bytes: number): string => {
    return `${Math.ceil(bytes / 1024 ** 3)} GB`;
};

const createTrainingDeviceName = (trainingDevice: TrainingDevice): string => {
    let name: string = trainingDevice.name;

    if (trainingDevice.memory != null) {
        const memory = formatTrainingDeviceMemory(trainingDevice.memory);
        name += ` (${memory})`;
    }

    if (trainingDevice.index != null) {
        name += ` [${trainingDevice.index}]`;
    }

    return name;
};

export const SelectTrainingDevice = () => {
    const { trainingDevices, onSelectTrainingDevice, selectedTrainingDevice } = useTrainModelState();

    return (
        <Picker
            flex={1}
            items={trainingDevices}
            label={'Select training device'}
            selectedKey={selectedTrainingDevice}
            onSelectionChange={(key) => key !== null && onSelectTrainingDevice(key.toString())}
        >
            {(item) => <Item key={createTrainingDeviceKey(item)}>{createTrainingDeviceName(item)}</Item>}
        </Picker>
    );
};
