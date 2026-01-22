// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Flex } from '@geti/ui';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { useTrainModel } from '../train-model-provider.component';
import { AllModelArchitectures } from './all-model-architectures.component';
import { RecommendedModelArchitectures } from './recommended-model-architectures.component';

const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureType[]) => {
    return modelArchitectures.slice(0, 3);
};

export const ModelArchitecturesList = () => {
    const [showMore, setShowMore] = useState<boolean>(false);
    const { activeModelArchitectureId, modelArchitectures, selectedModelArchitectureId, onSelectModelArchitectureId } =
        useTrainModel();

    if (showMore) {
        return (
            <Flex direction={'column'} minHeight={0} gap={'size-300'}>
                <AllModelArchitectures
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
                <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(false)}>
                    Show less
                </Button>
            </Flex>
        );
    }

    const recommendedArchitectures = getRecommendedArchitectures(modelArchitectures);

    return (
        <Flex direction={'column'} minHeight={0} gap={'size-300'}>
            <RecommendedModelArchitectures
                activeModelArchitectureId={activeModelArchitectureId}
                modelArchitectures={recommendedArchitectures}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
            />
            <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(true)}>
                Show more
            </Button>
        </Flex>
    );
};
