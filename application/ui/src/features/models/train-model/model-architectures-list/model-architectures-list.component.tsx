// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Button, Flex, Grid, RadioGroup } from '@geti/ui';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { SortModelArchitectures } from '../sort-model-architectures/sort-model-architectures.component';
import { SORT_OPTIONS, SORTING_HANDLERS, SortingOptions } from '../sort-model-architectures/utils';
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
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const ModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;
    const isActive = activeModelArchitectureId === modelArchitecture.id;

    return (
        <ModelArchitectureCard
            isCompact
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <Flex direction={'column'} width={'100%'} minWidth={0} gap={'size-100'}>
                {isActive && <ModelArchitectureCard.Active />}
                <Flex alignItems={'center'} justifyContent={'space-between'}>
                    <ModelArchitectureCard.Name />
                    <ModelArchitectureCard.Description />
                </Flex>
            </Flex>
            <ModelArchitectureCard.Parameters />
        </ModelArchitectureCard>
    );
};

interface AllModelArchitecturesProps {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const AllModelArchitectures = ({
    activeModelArchitectureId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: AllModelArchitecturesProps) => {
    const [sortBy, setSortBy] = useState<SortingOptions>(SortingOptions.RELEVANCE_ASC);
    const sortedModelArchitectures = SORTING_HANDLERS[sortBy](modelArchitectures);

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <SortModelArchitectures sortBy={sortBy} onSort={setSortBy} items={SORT_OPTIONS} />
            <ModelArchitecturesList
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                ariaLabel={'ALL model architectures'}
            >
                {sortedModelArchitectures.map((modelArchitecture) => (
                    <ModelArchitecture
                        key={modelArchitecture.id}
                        activeModelArchitectureId={activeModelArchitectureId}
                        modelArchitecture={modelArchitecture}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                ))}
            </ModelArchitecturesList>
        </Flex>
    );
};

interface RecommendedModelArchitectureProps {
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: RecommendedModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;
    const isActive = activeModelArchitectureId === modelArchitecture.id;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <Flex width={'100%'} minWidth={0} direction={'column'} gap={'size-100'}>
                {isActive && <ModelArchitectureCard.Active />}
                <ModelArchitectureCard.Name />
            </Flex>
            <ModelArchitectureCard.Parameters />
            <ModelArchitectureCard.Divider />
            <ModelArchitectureCard.ExpandedDescription />
        </ModelArchitectureCard>
    );
};

interface RecommendedModelArchitectures {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

const RecommendedModelArchitectures = ({
    activeModelArchitectureId,
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
                    activeModelArchitectureId={activeModelArchitectureId}
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
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
}

export const ModelArchitecturesListContainer = ({
    modelArchitectures,
    activeModelArchitectureId,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: ModelArchitecturesListContainer) => {
    const [showMore, setShowMore] = useState<boolean>(false);

    if (showMore) {
        return (
            <Flex direction={'column'} minHeight={0} gap={'size-300'}>
                <AllModelArchitectures
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
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
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            />
            <Button alignSelf={'start'} variant={'primary'} onPress={() => setShowMore(true)}>
                Show more
            </Button>
        </Flex>
    );
};
