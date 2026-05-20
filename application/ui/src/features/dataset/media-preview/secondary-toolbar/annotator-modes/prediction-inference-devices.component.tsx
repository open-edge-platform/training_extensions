// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { InferenceDevices } from '../../../../../components/inference-devices/inference-devices.component';
import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

type PredictionInferenceDevicesProps = {
    isDisabled?: boolean;
};

export const PredictionInferenceDevices = ({ isDisabled }: PredictionInferenceDevicesProps) => {
    const { selectedDevice, changeSelectedDevice } = usePredictionSetup();

    return (
        <InferenceDevices
            isQuiet
            selectedKey={selectedDevice}
            onSelectionChange={changeSelectedDevice}
            isDisabled={isDisabled}
        />
    );
};
