// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, use, useState } from 'react';

import {
    DatasetRevision,
    DeviceType,
    ModelArchitecture,
    RecommendedModelArchitectures,
    TrainingDevice,
} from '../../../constants/shared-types';
import { useGetActiveModelArchitectureId } from '../hooks/api/use-get-active-model-architecture-id.hook';
import { useGetDatasetRevisions } from '../hooks/api/use-get-dataset-revisions';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';

type TrainModelContextProps = {
    modelArchitectures: ModelArchitecture[];
    recommendedModelArchitectures: RecommendedModelArchitectures | null;

    activeModelArchitectureId: string | undefined;

    selectedModelArchitectureId: string | null;
    onSelectModelArchitectureId: (id: string | null) => void;

    trainingDevices: TrainingDevice[];
    selectedTrainingDevice: DeviceType | null;
    onSelectTrainingDevice: (deviceType: DeviceType | null) => void;

    datasetRevisions: DatasetRevision[];
    selectedDatasetRevision: string | null;
    onSelectDatasetRevision: (datasetRevision: string | null) => void;
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
};

export const TrainModelProvider = ({ children }: TrainModelProviderProps) => {
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

    return (
        <TrainModelContext
            value={{
                modelArchitectures: data.model_architectures,
                recommendedModelArchitectures: data.top_picks,

                activeModelArchitectureId,

                selectedModelArchitectureId,
                onSelectModelArchitectureId: setSelectedModelArchitectureId,

                trainingDevices,
                selectedTrainingDevice,
                onSelectTrainingDevice: setSelectedTrainingDevice,

                datasetRevisions,
                selectedDatasetRevision,
                onSelectDatasetRevision: setSelectedDatasetRevision,
            }}
        >
            {children}
        </TrainModelContext>
    );
};

export const useTrainModel = () => {
    const context = use(TrainModelContext);

    if (context === null) {
        throw new Error('useTrainModel must be used within a TrainModelProvider');
    }

    return context;
};
