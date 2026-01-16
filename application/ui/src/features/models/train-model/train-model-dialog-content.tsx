// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Divider, Flex, Loading, View } from '@geti/ui';

import { ModelArchitecturesList } from './model-architectures-list/model-architectures-list.component';
import { SelectDatasetRevision } from './select-dataset-revision.component';
import { SelectTrainingDevice } from './select-training-device.component';

interface TrainModelDialogContentProps {
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

export const TrainModelDialogContent = ({
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: TrainModelDialogContentProps) => {
    return (
        <View padding={'size-300'} backgroundColor={'gray-50'}>
            <Flex height={'100%'} direction={'column'} gap={'size-300'}>
                <Suspense fallback={<Loading mode={'inline'} />}>
                    <ModelArchitecturesList
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                </Suspense>

                <Divider size={'S'} width={'100%'} />
                <Flex gap={'size-300'} width={'100%'}>
                    <SelectTrainingDevice />
                    <SelectDatasetRevision />
                </Flex>
            </Flex>
        </View>
    );
};
