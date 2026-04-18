// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Flex } from '@geti/ui';

import { useTrainModelState } from '../train-model-provider.component';
import { AllModelArchitectures } from './all-model-architectures.component';
import { RecommendedModelArchitectures } from './recommended-model-architectures.component';
import { getRecommendedModelArchitecturesWithActiveArchitecture } from './utils';

const SHOW_MORE_THRESHOLD = 4;

export const ModelArchitecturesList = () => {
    const [showMore, setShowMore] = useState<boolean>(false);
    const { activeModelArchitectureId, modelArchitectures, selectedModelArchitectureId, onSelectModelArchitectureId } =
        useTrainModelState();

    const recommendedArchitectures = getRecommendedModelArchitecturesWithActiveArchitecture(
        modelArchitectures,
        activeModelArchitectureId
    );
    const collapsedArchitectures = recommendedArchitectures.slice(0, SHOW_MORE_THRESHOLD);
    const canToggleArchitecturesList = modelArchitectures.length > SHOW_MORE_THRESHOLD;

    return (
        <Flex direction={'column'} minHeight={0} gap={'size-300'}>
            {showMore ? (
                <AllModelArchitectures
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
            ) : (
                <RecommendedModelArchitectures
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={collapsedArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
            )}

            {canToggleArchitecturesList && (
                <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(!showMore)}>
                    {showMore ? 'Show less' : 'Show more'}
                </Button>
            )}
        </Flex>
    );
};
