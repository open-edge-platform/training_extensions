// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Button, Flex, Grid, RadioGroup } from '@geti/ui';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { useGetActiveModelId } from '../../hooks/api/use-get-active-model-id.hook';
import { useGetTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { ModelArchitectureCard } from './model-architecture/model-architecture.component';

import styles from './model-architectures-list.module.scss';

interface ModelArchitecturesListProps {
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
    ariaLabel: string;
    children: ReactNode;
}

const ModelArchitecturesList = ({
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
    children,
    ariaLabel,
}: ModelArchitecturesListProps) => {
    return (
        <RadioGroup
            isEmphasized
            onChange={onSelectedModelArchitectureIdChange}
            value={selectedModelArchitectureId}
            aria-label={ariaLabel}
        >
            <Grid UNSAFE_className={styles.gridLayout}>{children}</Grid>
        </RadioGroup>
    );
};

interface ModelArchitectureProps {
    activeModelId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const ModelArchitecture = ({
    activeModelId,
    modelArchitecture,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;

    return (
        <ModelArchitectureCard
            isCompact
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <ModelArchitectureCard.Name />
                <ModelArchitectureCard.Description />
            </Flex>
            <ModelArchitectureCard.Parameters />
        </ModelArchitectureCard>
    );
};

interface AllModelArchitecturesProps {
    activeModelId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const AllModelArchitectures = ({
    activeModelId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: AllModelArchitecturesProps) => {
    return (
        <ModelArchitecturesList
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Recommended model architectures'}
        >
            {modelArchitectures.map((modelArchitecture) => (
                <ModelArchitecture
                    key={modelArchitecture.id}
                    activeModelId={activeModelId}
                    modelArchitecture={modelArchitecture}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
            ))}
        </ModelArchitecturesList>
    );
};

interface RecommendedModelArchitectureProps {
    activeModelId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitecture = ({
    activeModelId,
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: RecommendedModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <ModelArchitectureCard.Name />
            <ModelArchitectureCard.Parameters />
            <ModelArchitectureCard.Divider />
            <ModelArchitectureCard.ExpandedDescription />
        </ModelArchitectureCard>
    );
};

interface RecommendedModelArchitectures {
    activeModelId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitectures = ({
    activeModelId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: RecommendedModelArchitectures) => {
    return (
        <ModelArchitecturesList
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Recommended model architectures'}
        >
            {modelArchitectures.map((modelArchitecture) => (
                <RecommendedModelArchitecture
                    key={modelArchitecture.id}
                    activeModelId={activeModelId}
                    modelArchitecture={modelArchitecture}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
            ))}
        </ModelArchitecturesList>
    );
};

const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureType[]) => {
    return modelArchitectures.slice(0, 3);
};

interface ModelArchitecturesListContainer {
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

export const ModelArchitecturesListContainer = ({
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: ModelArchitecturesListContainer) => {
    const { data } = useGetTaskModelArchitectures();
    const [showMore, setShowMore] = useState<boolean>(false);
    const activeModelId = useGetActiveModelId();

    if (showMore) {
        return (
            <Flex direction={'column'} minHeight={0} gap={'size-300'}>
                <AllModelArchitectures
                    activeModelId={activeModelId}
                    modelArchitectures={data.model_architectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
                <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(false)}>
                    Show less
                </Button>
            </Flex>
        );
    }

    const recommendedArchitectures = getRecommendedArchitectures(data.model_architectures);

    return (
        <Flex direction={'column'} minHeight={0} gap={'size-300'}>
            <RecommendedModelArchitectures
                activeModelId={activeModelId}
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
