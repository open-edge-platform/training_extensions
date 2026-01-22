// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, use, useState } from 'react';

import {
    DatasetRevision,
    DeviceType,
    ModelArchitecture,
    ModelArchitectureWithPerformanceCategory,
    RecommendedModelArchitectures,
    TrainingDevice,
} from '../../../constants/shared-types';
import { TrainingConfiguration } from '../configuration.interface';
import { useGetActiveModelArchitectureId } from '../hooks/api/use-get-active-model-architecture-id.hook';
import { useGetDatasetRevisions } from '../hooks/api/use-get-dataset-revisions';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingConfiguration } from '../hooks/api/use-get-training-configuration';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';

type TrainModelContextProps = {
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];

    activeModelArchitectureId: string | undefined;

    selectedModelArchitectureId: string | null;
    onSelectModelArchitectureId: (id: string | null) => void;

    trainingDevices: TrainingDevice[];
    selectedTrainingDevice: DeviceType | null;
    onSelectTrainingDevice: (deviceType: DeviceType | null) => void;

    datasetRevisions: DatasetRevision[];
    selectedDatasetRevision: string | null;
    onSelectDatasetRevision: (datasetRevision: string | null) => void;

    trainingConfiguration: TrainingConfiguration | undefined;
    onTrainingConfigurationChange: (trainingConfiguration: TrainingConfiguration) => void;
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
};

const getModelArchitectures = (
    modelArchitectures: ModelArchitecture[],
    recommendedModelArchitectures: RecommendedModelArchitectures | null
): ModelArchitectureWithPerformanceCategory[] => {
    if (recommendedModelArchitectures === null) {
        return modelArchitectures;
    }

    // Recommended architectures have the shape like { balance: "id-1", speed: "id-2", accuracy: "id-3" }
    // Here we need to convert it to { "id-1": "balance", "id-2": "speed", "id-3": "accuracy" }
    const recommendedArchitectureIdToCategory = Object.fromEntries(
        Object.entries(recommendedModelArchitectures).map(([key, value]) => [value, key])
    );

    return modelArchitectures.map((modelArchitecture) => {
        if (recommendedArchitectureIdToCategory[modelArchitecture.id] === undefined) {
            return modelArchitecture;
        }

        return {
            ...modelArchitecture,
            performanceCategory: recommendedArchitectureIdToCategory[modelArchitecture.id],
        };
    });
};

export const TrainModelProvider = ({ children }: TrainModelProviderProps) => {
    const { data } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const { data: datasetRevisions } = useGetDatasetRevisions();

    const activeModelArchitectureId = useGetActiveModelArchitectureId();

    const modelArchitectures: ModelArchitectureWithPerformanceCategory[] = getModelArchitectures(
        data.model_architectures,
        data.top_picks
    );

    const activeModelArchitecture = data.model_architectures.find(
        (modelArchitecture) => modelArchitecture.id === activeModelArchitectureId
    );

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(
        activeModelArchitecture?.id ?? null
    );

    const [trainingConfiguration, setTrainingConfiguration, defaultTrainingConfiguration] =
        useGetTrainingConfiguration(selectedModelArchitectureId);

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<DeviceType | null>(
        trainingDevices?.at(0)?.type ?? null
    );
    const [selectedDatasetRevision, setSelectedDatasetRevision] = useState<string | null>(
        datasetRevisions?.at(0)?.id ?? null
    );

    return (
        <TrainModelContext
            value={{
                modelArchitectures,

                activeModelArchitectureId,

                selectedModelArchitectureId,
                onSelectModelArchitectureId: setSelectedModelArchitectureId,

                trainingDevices,
                selectedTrainingDevice,
                onSelectTrainingDevice: setSelectedTrainingDevice,

                datasetRevisions,
                selectedDatasetRevision,
                onSelectDatasetRevision: setSelectedDatasetRevision,

                trainingConfiguration,
                onTrainingConfigurationChange: setTrainingConfiguration,
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
