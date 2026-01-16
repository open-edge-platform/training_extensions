// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Divider, Flex, Loading, View } from '@geti/ui';

import type { ModelArchitecture as ModelArchitectureType, TrainingDevices } from '../../../constants/shared-types';
import { ModelArchitecturesListContainer } from './model-architectures-list/model-architectures-list.component';
import { SelectDatasetRevision } from './select-dataset-revision.component';
import { SelectTrainingDevice } from './select-training-device.component';

interface TrainModelDialogContentProps {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;

    trainingDevices: TrainingDevices[];
    selectedTrainingDevice: string | null;
    onSelectedTrainingDeviceChange: (trainingDeviceId: string | null) => void;
}

export const TrainModelDialogContent = ({
    trainingDevices,
    onSelectedTrainingDeviceChange,
    selectedTrainingDevice,

    modelArchitectures,
    activeModelArchitectureId,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: TrainModelDialogContentProps) => {
    return (
        <View padding={'size-300'} backgroundColor={'gray-50'} height={'100%'}>
            <Flex height={'100%'} direction={'column'} gap={'size-300'}>
                <View flex={1} minHeight={0} overflow={'auto'}>
                    <ModelArchitecturesListContainer
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
                    <SelectDatasetRevision />
                </Flex>
            </Flex>
        </View>
    );
};
