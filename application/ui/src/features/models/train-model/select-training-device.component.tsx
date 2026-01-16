// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import type { TrainingDevices } from '../../../constants/shared-types';

interface SelectTrainingDeviceProps {
    trainingDevices: TrainingDevices[];
    selectedTrainingDevice: string | null;
    onSelectedTrainingDeviceChange: (trainingDeviceId: string | null) => void;
}

export const SelectTrainingDevice = ({
    trainingDevices,
    onSelectedTrainingDeviceChange,
    selectedTrainingDevice,
}: SelectTrainingDeviceProps) => {
    return (
        <Picker
            items={trainingDevices}
            label={'Select training device'}
            selectedKey={selectedTrainingDevice}
            onSelectionChange={(key) => onSelectedTrainingDeviceChange(String(key))}
        >
            {(item) => <Item key={item.type}>{item.name}</Item>}
        </Picker>
    );
};
