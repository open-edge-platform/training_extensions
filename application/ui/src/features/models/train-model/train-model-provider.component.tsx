// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, use, useMemo, useState } from 'react';

import {
    DatasetRevision,
    DeviceType,
    Model,
    ModelArchitectureWithPerformanceCategory,
    TrainingDevice,
} from '../../../constants/shared-types';
import { useGetDatasetRevisions } from '../../../hooks/use-get-dataset-revisions.hook';
import { useGetActiveModel } from '../hooks/api/use-get-active-model.hook';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetModels } from '../hooks/api/use-get-models.hook';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';

type DatasetRevisionWithValue = Pick<DatasetRevision, 'id' | 'name'> & { value: string | null };
type ModelRevisionWithValue = Pick<Model, 'id' | 'name' | 'architecture'> & { value: string | null };

type TrainModelContextProps = {
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];

    activeModelArchitectureId: string | undefined;

    selectedModelArchitectureId: string | null;
    onSelectModelArchitectureId: (id: string | null) => void;

    trainingDevices: TrainingDevice[];
    selectedTrainingDevice: DeviceType | null;
    onSelectTrainingDevice: (deviceType: DeviceType | null) => void;

    datasetRevisions: DatasetRevisionWithValue[];
    selectedDatasetRevisionId: string | null;
    onSelectDatasetRevisionId: (datasetRevision: string | null) => void;

    modelRevisions: ModelRevisionWithValue[];
    selectedModelRevisionId: string | null;
    onSelectModelRevisionId: (modelRevisionId: string | null) => void;
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
    preSelectedDatasetRevisionId?: string;
    preSelectedModelRevisionId?: string;
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

const useModelRevisions = () => {
    const { data: models } = useGetModels();

    return {
        modelRevisions: [
            { id: 'train-from-scratch', name: 'Train from scratch', architecture: '', value: null },
            ...(models?.map(({ id, name, architecture }) => ({ id, name, architecture, value: String(id) })) ?? []),
        ],
    };
};

const getModelRevisionsForArchitecture = (
    modelRevisions: ModelRevisionWithValue[],
    architectureId: string | null
): ModelRevisionWithValue[] => {
    return modelRevisions.filter((modelRevision) => {
        if (modelRevision.id === 'train-from-scratch') {
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
    const firstRevision = revisionsForArchitecture.find(({ id }) => id !== 'train-from-scratch');

    return firstRevision?.id ?? revisionsForArchitecture.at(0)?.id ?? null;
};

export const TrainModelProvider = ({
    children,
    preSelectedDatasetRevisionId,
    preSelectedModelRevisionId,
}: TrainModelProviderProps) => {
    const { modelArchitectures } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const { datasetRevisions } = useDatasetRevisions();
    const { modelRevisions: allModelRevisions } = useModelRevisions();
    const activeModel = useGetActiveModel();
    const preSelectedModelRevision = allModelRevisions.find(({ id }) => id === preSelectedModelRevisionId);

    const activeModelArchitecture = modelArchitectures.find(
        (modelArchitecture) => modelArchitecture.id === activeModel?.architecture
    );

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(
        preSelectedModelRevision?.architecture ?? activeModelArchitecture?.id ?? null
    );

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<DeviceType | null>(
        trainingDevices?.at(0)?.type ?? null
    );
    const [selectedDatasetRevisionId, setSelectedDatasetRevisionId] = useState<string | null>(
        preSelectedDatasetRevisionId ?? datasetRevisions?.at(0)?.id ?? null
    );
    const [selectedModelRevisionId, setSelectedModelRevisionId] = useState<string | null>(
        () =>
            preSelectedModelRevisionId ??
            getDefaultModelRevisionIdForArchitecture(allModelRevisions, selectedModelArchitectureId)
    );

    const modelRevisions = useMemo(() => {
        return getModelRevisionsForArchitecture(allModelRevisions, selectedModelArchitectureId);
    }, [allModelRevisions, selectedModelArchitectureId]);

    const onSelectModelArchitectureId = (modelArchitectureId: string | null) => {
        setSelectedModelArchitectureId(modelArchitectureId);
        setSelectedModelRevisionId(getDefaultModelRevisionIdForArchitecture(allModelRevisions, modelArchitectureId));
    };

    return (
        <TrainModelContext
            value={{
                modelArchitectures,

                activeModelArchitectureId: activeModel?.architecture,

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
