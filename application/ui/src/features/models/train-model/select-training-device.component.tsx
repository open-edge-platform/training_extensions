// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import type { DeviceType } from '../../../constants/shared-types';
import { useTrainModel } from './train-model-provider.component';

export const SelectTrainingDevice = () => {
    const { trainingDevices, onSelectTrainingDevice, selectedTrainingDevice } = useTrainModel();

    return (
        <Picker
            flex={1}
            items={trainingDevices}
            label={'Select training device'}
            selectedKey={selectedTrainingDevice}
            onSelectionChange={(key) => onSelectTrainingDevice(key as DeviceType)}
        >
            {(item) => <Item key={item.type}>{item.name}</Item>}
        </Picker>
    );
};
