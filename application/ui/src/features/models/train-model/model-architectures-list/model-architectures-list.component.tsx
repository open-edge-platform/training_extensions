// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Button, Flex } from '@geti/ui';

import { useTrainModel } from '../train-model-provider.component';
import { AllModelArchitectures } from './all-model-architectures.component';
import { RecommendedModelArchitectures } from './recommended-model-architectures.component';
import { getRecommendedModelArchitecturesWithActiveArchitecture } from './utils';

type ModelArchitecturesContainerProps = {
    children: ReactNode;
    showMore: boolean;
    onShowMore: (showMore: boolean) => void;
};

const ModelArchitecturesContainer = ({ children, onShowMore, showMore }: ModelArchitecturesContainerProps) => {
    return (
        <Flex direction={'column'} minHeight={0} gap={'size-300'}>
            {children}

            <Button alignSelf={'start'} variant={'primary'} onPress={() => onShowMore(!showMore)}>
                {showMore ? 'Show less' : 'Show more'}
            </Button>
        </Flex>
    );
};

export const ModelArchitecturesList = () => {
    const [showMore, setShowMore] = useState<boolean>(false);
    const { activeModelArchitectureId, modelArchitectures, selectedModelArchitectureId, onSelectModelArchitectureId } =
        useTrainModel();

    if (showMore) {
        return (
            <ModelArchitecturesContainer showMore={showMore} onShowMore={setShowMore}>
                <AllModelArchitectures
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
            </ModelArchitecturesContainer>
        );
    }

    const recommendedArchitectures = getRecommendedModelArchitecturesWithActiveArchitecture(
        modelArchitectures,
        activeModelArchitectureId
    );

    return (
        <ModelArchitecturesContainer showMore={showMore} onShowMore={setShowMore}>
            <RecommendedModelArchitectures
                activeModelArchitectureId={activeModelArchitectureId}
                modelArchitectures={recommendedArchitectures}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
            />
        </ModelArchitecturesContainer>
    );
};
