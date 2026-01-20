// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, View } from '@geti/ui';

import type {
    DatasetRevision,
    DeviceType,
    ModelArchitecture as ModelArchitectureType,
    TrainingDevices,
} from '../../../constants/shared-types';
import { ModelArchitecturesList } from './model-architectures-list/model-architectures-list.component';
import { SelectDatasetRevision } from './select-dataset-revision.component';
import { SelectTrainingDevice } from './select-training-device.component';

type TrainModelDialogContentProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;

    trainingDevices: TrainingDevices[];
    selectedTrainingDevice: DeviceType | null;
    onSelectedTrainingDeviceChange: (trainingDeviceId: DeviceType | null) => void;

    datasetRevisions: DatasetRevision[];
    selectedDatasetRevision: string | null;
    onSelectedDatasetRevisionChange: (datasetRevisionId: string | null) => void;
};

export const TrainModelDialogContent = ({
    trainingDevices,
    onSelectedTrainingDeviceChange,
    selectedTrainingDevice,

    datasetRevisions,
    onSelectedDatasetRevisionChange,
    selectedDatasetRevision,

    modelArchitectures,
    activeModelArchitectureId,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: TrainModelDialogContentProps) => {
    return (
        <View padding={'size-300'} backgroundColor={'gray-50'} height={'100%'}>
            <Flex height={'100%'} direction={'column'} gap={'size-300'}>
                <View flex={1} minHeight={0} overflow={'auto'}>
                    <ModelArchitecturesList
                        modelArchitectures={modelArchitectures}
                        activeModelArchitectureId={activeModelArchitectureId}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                </View>

                <Divider size={'S'} width={'100%'} />
                <Flex gap={'size-300'} width={'100%'}>
                    <SelectTrainingDevice
                        trainingDevices={trainingDevices}
                        selectedTrainingDevice={selectedTrainingDevice}
                        onSelectedTrainingDeviceChange={onSelectedTrainingDeviceChange}
                    />
                    <SelectDatasetRevision
                        datasetRevisions={datasetRevisions}
                        selectedDatasetRevision={selectedDatasetRevision}
                        onSelectedDatasetRevisionChange={onSelectedDatasetRevisionChange}
                    />
                </Flex>
            </Flex>
        </View>
    );
};
