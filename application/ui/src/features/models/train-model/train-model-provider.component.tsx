// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, use, useState } from 'react';

import {
    DatasetRevision,
    DeviceType,
    ModelArchitectureWithPerformanceCategory,
    TrainingDevice,
} from '../../../constants/shared-types';
import { useGetDatasetRevisions } from '../../../hooks/use-get-dataset-revisions.hook';
import { useGetActiveModel } from '../hooks/api/use-get-active-model.hook';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';

type DatasetRevisionWithValue = Pick<DatasetRevision, 'id' | 'name'> & { value: string | null };

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
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
    preSelectedDatasetRevisionId?: string;
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

export const TrainModelProvider = ({ children, preSelectedDatasetRevisionId }: TrainModelProviderProps) => {
    const { modelArchitectures } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const { datasetRevisions } = useDatasetRevisions();
    const activeModel = useGetActiveModel();

    const activeModelArchitecture = modelArchitectures.find(
        (modelArchitecture) => modelArchitecture.id === activeModel?.architecture
    );

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(
        activeModelArchitecture?.id ?? null
    );

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<DeviceType | null>(
        trainingDevices?.at(0)?.type ?? null
    );
    const [selectedDatasetRevisionId, setSelectedDatasetRevisionId] = useState<string | null>(
        preSelectedDatasetRevisionId ?? datasetRevisions?.at(0)?.id ?? null
    );

    return (
        <TrainModelContext
            value={{
                modelArchitectures,

                activeModelArchitectureId: activeModel?.architecture,

                selectedModelArchitectureId,
                onSelectModelArchitectureId: setSelectedModelArchitectureId,

                trainingDevices,
                selectedTrainingDevice,
                onSelectTrainingDevice: setSelectedTrainingDevice,

                datasetRevisions,
                selectedDatasetRevisionId,
                onSelectDatasetRevisionId: setSelectedDatasetRevisionId,
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
