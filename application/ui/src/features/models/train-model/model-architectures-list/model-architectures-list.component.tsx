// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Button, Flex } from '@geti/ui';

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { useTrainModel } from '../train-model-provider.component';
import { AllModelArchitectures } from './all-model-architectures.component';
import { RecommendedModelArchitectures } from './recommended-model-architectures.component';

const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureWithPerformanceCategory[]) => {
    const recommended = modelArchitectures.filter(
        (modelArchitecture) => modelArchitecture.performanceCategory !== undefined
    );

    if (recommended.length > 0) {
        return recommended;
    }

    // For now just return top 3 recommended architectures, but in the future we can add more logic here
    return modelArchitectures.slice(0, 3);
};

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

    const recommendedArchitectures = getRecommendedArchitectures(modelArchitectures);

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
