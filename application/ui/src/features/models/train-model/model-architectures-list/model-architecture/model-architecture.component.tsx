// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext } from 'react';

import { Content, ContextualHelp, Divider, Heading, Radio, Text, View } from '@geti/ui';
import { clsx } from 'clsx';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../../constants/shared-types';

import styles from './model-architecture.module.scss';

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

const ModelArchitectureExpandedDescription = () => {
    const { modelArchitecture } = useModelArchitecture();

    return <Text UNSAFE_className={styles.modelArchitectureExpandedDescription}>{modelArchitecture.description}</Text>;
};

const ModelArchitectureDescription = () => {
    const { modelArchitecture } = useModelArchitecture();

    return (
        <ContextualHelp variant='info'>
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
            <li>Number of parameters: {modelArchitecture.stats.trainable_parameters} Millions</li>
            <li>License: Apache 2.0</li>
        </ul>
    );
};

const ModelArchitectureName = () => {
    const { modelArchitecture, isSelected } = useModelArchitecture();

    return (
        <Radio
            value={modelArchitecture.id}
            UNSAFE_className={clsx(styles.modelArchitectureName, {
                [styles.modelArchitectureNameSelected]: isSelected,
            })}
        >
            {modelArchitecture.name}
        </Radio>
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
    isCompact?: boolean;
    children: ReactNode;
    onSelect: () => void;
    modelArchitecture: ModelArchitectureType;
};

export const ModelArchitectureCard = ({
    isSelected,
    isCompact,
    children,
    onSelect,
    modelArchitecture,
}: ModelArchitectureProps) => {
    return (
        <ModelArchitectureContext value={{ isSelected, modelArchitecture }}>
            <div
                className={clsx(styles.modelArchitectureContainer, {
                    [styles.modelArchitectureSelected]: isSelected,
                    [styles.modelArchitectureCompact]: isCompact,
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
ModelArchitectureCard.ExpandedDescription = ModelArchitectureExpandedDescription;
ModelArchitectureCard.Description = ModelArchitectureDescription;
ModelArchitectureCard.Active = ActiveModelArchitecture;
