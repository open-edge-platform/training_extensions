// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { DeviceType } from '../../../constants/shared-types';
import { useGetActiveModelArchitectureId } from '../hooks/api/use-get-active-model-architecture-id.hook';
import { useGetDatasetRevisions } from '../hooks/api/use-get-dataset-revisions';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';

export const useTrainModel = () => {
    const { data } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const { data: datasetRevisions } = useGetDatasetRevisions();
    const activeModelArchitectureId = useGetActiveModelArchitectureId();

    const activeModelArchitecture = data.model_architectures.find(
        (modelArchitecture) => modelArchitecture.id === activeModelArchitectureId
    );

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(
        activeModelArchitecture?.id ?? null
    );

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<DeviceType | null>(
        trainingDevices?.at(0)?.type ?? null
    );
    const [selectedDatasetRevision, setSelectedDatasetRevision] = useState<string | null>(
        datasetRevisions?.at(0)?.id ?? null
    );

    const isStartButtonDisabled =
        selectedModelArchitectureId === null || selectedTrainingDevice === null || selectedDatasetRevision === null;

    return {
        modelArchitectures: data.model_architectures,
        trainingDevices,
        datasetRevisions,
        activeModelArchitectureId,

        selectedModelArchitectureId,
        onSelectedModelArchitectureIdChange: setSelectedModelArchitectureId,

        selectedTrainingDevice,
        onSelectedTrainingDeviceChange: setSelectedTrainingDevice,

        selectedDatasetRevision,
        onSelectedDatasetRevisionChange: setSelectedDatasetRevision,

        isStartButtonDisabled,
    };
};
