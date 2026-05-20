// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import { createDeviceName } from '../../../../components/util';
import { createTrainingDeviceKey, useTrainModelState } from '../train-model-provider.component';

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
            {(item) => <Item key={createTrainingDeviceKey(item)}>{createDeviceName(item)}</Item>}
        </Picker>
    );
};
