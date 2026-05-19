// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, use, useMemo, useState } from 'react';

import { useGetDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';

import {
    DatasetRevision,
    Model,
    ModelArchitectureWithPerformanceCategory,
    TrainingConfiguration,
    TrainingDevice,
} from '../../../constants/shared-types';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetSuccessfulModels } from '../hooks/api/use-get-models.hook';
import { useGetTrainingDevices } from './api/use-get-training-devices';
import { useTrainingConfiguration } from './hooks/use-training-configuration';
import { getDefaultTrainingDevice } from './select-training-device/utils';

type DatasetRevisionWithValue = Pick<DatasetRevision, 'id' | 'name'> & { value: string | null };
type ModelRevisionWithValue = Pick<Model, 'id' | 'name' | 'architecture'> & { value: string | null };

export type TrainModelContextProps = {
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];

    selectedModelArchitectureId: string | null;
    onSelectModelArchitectureId: (id: string | null) => void;

    trainingDevices: TrainingDevice[];
    selectedTrainingDevice: string | null;
    onSelectTrainingDevice: (deviceKey: string | null) => void;

    datasetRevisions: DatasetRevisionWithValue[];
    selectedDatasetRevisionId: string | null;
    onSelectDatasetRevisionId: (datasetRevision: string | null) => void;

    modelRevisions: ModelRevisionWithValue[];
    selectedModelRevisionId: string | null;
    onSelectModelRevisionId: (modelRevisionId: string | null) => void;

    isAdvancedSettingsMode: boolean;
    onToggleAdvancedSettingsMode: (isAdvancedSettingsMode: boolean) => void;

    trainingConfiguration: TrainingConfiguration | undefined;
    defaultTrainingConfiguration: TrainingConfiguration | undefined;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
};

const useDatasetRevisions = () => {
    const { data: datasetRevisions } = useGetDatasetRevisions();

    return {
        datasetRevisions: [
            { id: 'use-current-dataset-revision', name: 'Use current dataset', value: null },
            ...(datasetRevisions?.map(({ id, name }) => ({ id, name, value: String(id) })) ?? []),
        ],
    };
};

const TRAIN_FROM_SCRATCH = 'train-from-scratch';
const useModelRevisions = () => {
    const { data: models } = useGetSuccessfulModels();

    return {
        modelRevisions: [
            { id: TRAIN_FROM_SCRATCH, name: 'Train from scratch', architecture: '', value: null },
            ...(models?.map(({ id, name, architecture }) => ({ id, name, architecture, value: String(id) })) ?? []),
        ],
    };
};

const getModelRevisionsForArchitecture = (
    modelRevisions: ModelRevisionWithValue[],
    architectureId: string | null
): ModelRevisionWithValue[] => {
    return modelRevisions.filter((modelRevision) => {
        if (modelRevision.id === TRAIN_FROM_SCRATCH) {
            return true;
        }

        return modelRevision.architecture === architectureId;
    });
};

const getDefaultModelRevisionIdForArchitecture = (
    modelRevisions: ModelRevisionWithValue[],
    architectureId: string | null
): string | null => {
    const revisionsForArchitecture = getModelRevisionsForArchitecture(modelRevisions, architectureId);
    const firstRevision = revisionsForArchitecture.find(({ id }) => id !== TRAIN_FROM_SCRATCH);

    return firstRevision?.id ?? revisionsForArchitecture.at(0)?.id ?? null;
};

export const createTrainingDeviceKey = (trainingDevice: TrainingDevice): string => {
    if (trainingDevice.index == null) {
        return trainingDevice.type;
    }

    return `${trainingDevice.type}-${trainingDevice.index}`;
};

export const TrainModelProvider = ({ children }: TrainModelProviderProps) => {
    const { modelArchitectures } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const { datasetRevisions } = useDatasetRevisions();
    const { modelRevisions: allModelRevisions } = useModelRevisions();

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(null);

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<string | null>(() => {
        const defaultDevice = getDefaultTrainingDevice(trainingDevices);
        return defaultDevice ? createTrainingDeviceKey(defaultDevice) : null;
    });
    const [selectedDatasetRevisionId, setSelectedDatasetRevisionId] = useState<string | null>(
        datasetRevisions?.at(0)?.id ?? null
    );
    const [selectedModelRevisionId, setSelectedModelRevisionId] = useState<string | null>(() =>
        getDefaultModelRevisionIdForArchitecture(allModelRevisions, selectedModelArchitectureId)
    );

    const [isAdvancedSettingsMode, setIsAdvancedSettingsMode] = useState<boolean>(false);

    const modelRevisions = useMemo(() => {
        return getModelRevisionsForArchitecture(allModelRevisions, selectedModelArchitectureId);
    }, [allModelRevisions, selectedModelArchitectureId]);

    const selectedModelRevision = modelRevisions.find((modelRevision) => modelRevision.id === selectedModelRevisionId);

    const [trainingConfiguration, setTrainingConfiguration, defaultTrainingConfiguration] = useTrainingConfiguration({
        modelArchitectureId: selectedModelArchitectureId,
        modelRevisionId: selectedModelRevision?.value ?? null,
    });

    const onSelectModelArchitectureId = (modelArchitectureId: string | null) => {
        setSelectedModelArchitectureId(modelArchitectureId);
        setSelectedModelRevisionId(getDefaultModelRevisionIdForArchitecture(allModelRevisions, modelArchitectureId));
    };

    return (
        <TrainModelContext
            value={{
                modelArchitectures,

                selectedModelArchitectureId,
                onSelectModelArchitectureId,

                trainingDevices,
                selectedTrainingDevice,
                onSelectTrainingDevice: setSelectedTrainingDevice,

                datasetRevisions,
                selectedDatasetRevisionId,
                onSelectDatasetRevisionId: setSelectedDatasetRevisionId,

                modelRevisions,
                selectedModelRevisionId,
                onSelectModelRevisionId: setSelectedModelRevisionId,

                isAdvancedSettingsMode,
                onToggleAdvancedSettingsMode: setIsAdvancedSettingsMode,

                trainingConfiguration,
                defaultTrainingConfiguration,
                onTrainingConfigurationChange: setTrainingConfiguration,
            }}
        >
            {children}
        </TrainModelContext>
    );
};

export const useTrainModelState = () => {
    const context = use(TrainModelContext);

    if (context === null) {
        throw new Error('useTrainModel must be used within a TrainModelProvider');
    }

    return context;
};
