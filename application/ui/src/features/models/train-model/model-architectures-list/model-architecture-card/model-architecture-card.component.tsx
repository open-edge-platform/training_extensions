// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext } from 'react';

import { Content, ContextualHelp, Divider, Flex, Heading, Radio, Text, View } from '@geti/ui';
import { clsx } from 'clsx';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../../constants/shared-types';

import styles from './model-architecture-card.module.scss';

const ActiveModelArchitecture = () => {
    return (
        <View
            alignSelf={'start'}
            UNSAFE_className={styles.activeModelArchitecture}
            paddingX={'size-50'}
            borderRadius={'regular'}
        >
            <Text>Active model</Text>
        </View>
    );
};

const ModelArchitectureDescription = () => {
    const { modelArchitecture, isSelected } = useModelArchitecture();

    return (
        <ContextualHelp variant='info' UNSAFE_className={clsx({ [styles.description]: isSelected })}>
            <Heading>{modelArchitecture.name}</Heading>
            <Content>
                <Text>{modelArchitecture.description}</Text>
            </Content>
        </ContextualHelp>
    );
};

const ModelArchitectureDivider = () => {
    return <Divider size={'S'} />;
};

const ModelArchitectureParameters = () => {
    const { modelArchitecture } = useModelArchitecture();

    return (
        <ul className={styles.modelArchitectureParameters}>
            <li>Size: {modelArchitecture.stats.trainable_parameters} Millions</li>
            <li>Accuracy: {modelArchitecture.stats.performance_ratings.accuracy}</li>
            <li>Inference speed: {modelArchitecture.stats.performance_ratings.inference_speed}</li>
            <li>Training time: {modelArchitecture.stats.performance_ratings.training_time}</li>
            <li>License: Apache 2.0</li>
        </ul>
    );
};

const ModelArchitectureName = () => {
    const { modelArchitecture, isSelected } = useModelArchitecture();

    return (
        <Flex justifyContent={'space-between'} alignItems={'center'} minWidth={0}>
            <Radio
                flex={1}
                minWidth={0}
                value={modelArchitecture.id}
                UNSAFE_className={clsx(styles.modelArchitectureName, {
                    [styles.modelArchitectureNameSelected]: isSelected,
                })}
            >
                {modelArchitecture.name}
            </Radio>
            <ModelArchitectureDescription />
        </Flex>
    );
};

type ModelArchitectureContextProps = {
    isSelected: boolean;
    modelArchitecture: ModelArchitectureType;
};

const ModelArchitectureContext = createContext<ModelArchitectureContextProps | null>(null);

export const useModelArchitecture = () => {
    const context = useContext(ModelArchitectureContext);

    if (context === null) {
        throw new Error('useModelArchitecture was used outside of ModelArchitectureProvider');
    }

    return context;
};

type ModelArchitectureProps = {
    isSelected: boolean;
    children: ReactNode;
    onSelect: () => void;
    modelArchitecture: ModelArchitectureType;
};

export const ModelArchitectureCard = ({
    isSelected,
    children,
    onSelect,
    modelArchitecture,
}: ModelArchitectureProps) => {
    return (
        <ModelArchitectureContext value={{ isSelected, modelArchitecture }}>
            <div
                className={clsx(styles.modelArchitectureContainer, {
                    [styles.modelArchitectureSelected]: isSelected,
                })}
                onClick={onSelect}
            >
                {children}
            </div>
        </ModelArchitectureContext>
    );
};

ModelArchitectureCard.Name = ModelArchitectureName;
ModelArchitectureCard.Parameters = ModelArchitectureParameters;
ModelArchitectureCard.Divider = ModelArchitectureDivider;
ModelArchitectureCard.Description = ModelArchitectureDescription;
ModelArchitectureCard.Active = ActiveModelArchitecture;
