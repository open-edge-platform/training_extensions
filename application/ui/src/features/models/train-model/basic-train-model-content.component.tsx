// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, View } from '@geti/ui';

import { ModelArchitecturesList } from './model-architectures-list/model-architectures-list.component';
import { SelectDatasetRevision } from './select-dataset-revision.component';
import { SelectModelRevision } from './select-model-revision.component';
import { SelectTrainingDevice } from './select-training-device/select-training-device.component';

export const BasicTrainModelContent = () => {
    return (
        <View backgroundColor={'gray-50'} height={'100%'}>
            <Flex height={'100%'} direction={'column'} gap={'size-300'}>
                <View flex={1} minHeight={0} overflow={'auto'}>
                    <ModelArchitecturesList />
                </View>

                <Divider size={'S'} width={'100%'} />
                <Flex gap={'size-300'} width={'100%'}>
                    <SelectTrainingDevice />
                    <SelectDatasetRevision />
                    <SelectModelRevision />
                </Flex>
            </Flex>
        </View>
    );
};
