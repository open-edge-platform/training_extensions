// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext } from 'react';

import { Divider, Radio, Text } from '@geti/ui';
import { clsx } from 'clsx';

import { type ModelArchitecture as ModelArchitectureType } from '../../../../../constants/shared-types';

import styles from './model-architecture.module.scss';

const ModelArchitectureExpandedDescription = () => {
    const { modelArchitecture } = useModelArchitecture();

    return <Text UNSAFE_className={styles.modelArchitectureExpandedDescription}>{modelArchitecture.description}</Text>;
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

interface ModelArchitectureContextProps {
    isSelected: boolean;
    modelArchitecture: ModelArchitectureType;
}

const ModelArchitectureContext = createContext<ModelArchitectureContextProps | null>(null);

export const useModelArchitecture = () => {
    const context = useContext(ModelArchitectureContext);

    if (context === null) {
        throw new Error('useModelArchitecture was used outside of ModelArchitectureProvider');
    }

    return context;
};

interface ModelArchitectureProps {
    isSelected: boolean;
    children: ReactNode;
    onSelect: () => void;
    modelArchitecture: ModelArchitectureType;
}

export const ModelArchitecture = ({ isSelected, children, onSelect, modelArchitecture }: ModelArchitectureProps) => {
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

ModelArchitecture.Name = ModelArchitectureName;
ModelArchitecture.Parameters = ModelArchitectureParameters;
ModelArchitecture.Divider = ModelArchitectureDivider;
ModelArchitecture.ExpandedDescription = ModelArchitectureExpandedDescription;
