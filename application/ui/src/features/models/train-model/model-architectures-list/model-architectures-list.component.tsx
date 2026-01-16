// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Flex, Grid, RadioGroup } from '@geti/ui';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { useGetTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { ModelArchitecture } from './model-architecture/model-architecture.component';

import styles from './model-architectures-list.module.scss';

interface RecommendedModelArchitectureProps {
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitecture = ({
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: RecommendedModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;

    return (
        <ModelArchitecture
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <ModelArchitecture.Name name={modelArchitecture.name} isSelected={isSelected} id={modelArchitecture.id} />
            <ModelArchitecture.Parameters numberOfParameters={modelArchitecture.stats.trainable_parameters} />
            <ModelArchitecture.Divider />
            <ModelArchitecture.ExpandedDescription description={modelArchitecture.description} />
        </ModelArchitecture>
    );
};

interface RecommendedModelArchitectures {
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitectures = ({
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: RecommendedModelArchitectures) => {
    return (
        <RadioGroup
            flex={1}
            isEmphasized
            onChange={onSelectedModelArchitectureIdChange}
            value={selectedModelArchitectureId}
            aria-label={'Recommended model architectures'}
        >
            <Grid UNSAFE_className={styles.gridLayout}>
                {modelArchitectures.map((modelArchitecture) => (
                    <RecommendedModelArchitecture
                        modelArchitecture={modelArchitecture}
                        key={modelArchitecture.id}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                ))}
            </Grid>
        </RadioGroup>
    );
};

const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureType[]) => {
    return modelArchitectures.slice(0, 3);
};

interface ModelArchitecturesList {
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

export const ModelArchitecturesList = ({
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: ModelArchitecturesList) => {
    const { data } = useGetTaskModelArchitectures();
    const [showMore, setShowMore] = useState<boolean>(false);

    if (showMore) {
        return (
            <Grid>
                <></>
            </Grid>
        );
    }

    const recommendedArchitectures = getRecommendedArchitectures(data.model_architectures);

    return (
        <Flex direction={'column'} flex={1} minHeight={0} gap={'size-300'}>
            <RecommendedModelArchitectures
                modelArchitectures={recommendedArchitectures}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            />
            <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(true)}>
                Show more
            </Button>
        </Flex>
    );
};
