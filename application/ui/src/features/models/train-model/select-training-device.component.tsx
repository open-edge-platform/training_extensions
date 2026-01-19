// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import type { DeviceType, TrainingDevices } from '../../../constants/shared-types';

type SelectTrainingDeviceProps = {
    trainingDevices: TrainingDevices[];
    selectedTrainingDevice: DeviceType | null;
    onSelectedTrainingDeviceChange: (trainingDeviceId: DeviceType | null) => void;
};

export const SelectTrainingDevice = ({
    trainingDevices,
    onSelectedTrainingDeviceChange,
    selectedTrainingDevice,
}: SelectTrainingDeviceProps) => {
    return (
        <Picker
            flex={1}
            items={trainingDevices}
            label={'Select training device'}
            selectedKey={selectedTrainingDevice}
            onSelectionChange={(key) => onSelectedTrainingDeviceChange(key as DeviceType)}
        >
            {(item) => <Item key={item.type}>{item.name}</Item>}
        </Picker>
    );
};
