// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Loading } from '@geti/ui';

import { InferenceDevices } from '../../../../../components/inference-devices/inference-devices.component';
import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

type PredictionInferenceDevicesProps = {
    isDisabled?: boolean;
};

export const PredictionInferenceDevices = ({ isDisabled }: PredictionInferenceDevicesProps) => {
    const { selectedDevice, changeSelectedDevice } = usePredictionSetup();

    return (
        <Suspense fallback={<Loading mode={'inline'} />}>
            <InferenceDevices
                isQuiet
                ariaLabel={'Inference devices'}
                selectedKey={selectedDevice}
                onSelectionChange={changeSelectedDevice}
                isDisabled={isDisabled}
            />
        </Suspense>
    );
};
