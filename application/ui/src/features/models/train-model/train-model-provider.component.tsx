// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, use, useState } from 'react';

import { isEqual } from 'lodash-es';

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
import { useGetModelsQuery } from '../hooks/api/use-get-models.hook';
import { useGetTrainingConfiguration } from '../hooks/api/use-get-training-configuration';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';
import { areSubsetsSizesValid } from './advanced-settings/data-management/training-subsets/utils';

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
    defaultTrainingConfiguration: TrainingConfiguration | undefined;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;

    isReshufflingSubsetsEnabled: boolean;
    onReshufflingSubsetsEnabledChange: (reshufflingSubsetsEnabled: boolean) => void;

    trainFromScratch: boolean;
    onTrainFromScratchChange: (trainFromScratch: boolean) => void;

    hasSupportedModels: boolean;

    isValidConfiguration: (isAdvancedMode: boolean) => boolean;
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

const useModelsByArchitectureId = (modelArchitectureId: string | null) => {
    const { data } = useGetModelsQuery();

    return (data ?? []).filter((model) => model.architecture === modelArchitectureId);
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

    const [isReshufflingSubsetsEnabled, setIsReshufflingSubsetsEnabled] = useState<boolean>(false);
    const [trainFromScratch, setTrainFromScratch] = useState<boolean>(false);

    const modelsByArchitecture = useModelsByArchitectureId(selectedModelArchitectureId);

    const hasSupportedModels = modelsByArchitecture.length > 0;

    const isValidConfiguration = (isAdvancedMode: boolean) => {
        if (
            selectedModelArchitectureId === null ||
            selectedTrainingDevice === null ||
            selectedDatasetRevision === null
        ) {
            return false;
        }

        if (!isAdvancedMode) {
            return true;
        }

        if (trainingConfiguration === undefined || defaultTrainingConfiguration === undefined) {
            return false;
        }

        if (
            !isEqual(
                trainingConfiguration.dataset_preparation.subset_split,
                defaultTrainingConfiguration.dataset_preparation.subset_split
            )
        ) {
            return areSubsetsSizesValid(trainingConfiguration.dataset_preparation.subset_split);
        }

        return true;
    };

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
                defaultTrainingConfiguration,
                onUpdateTrainingConfiguration: setTrainingConfiguration,

                trainFromScratch,
                onTrainFromScratchChange: setTrainFromScratch,

                isReshufflingSubsetsEnabled,
                onReshufflingSubsetsEnabledChange: setIsReshufflingSubsetsEnabled,

                hasSupportedModels,

                isValidConfiguration,
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
