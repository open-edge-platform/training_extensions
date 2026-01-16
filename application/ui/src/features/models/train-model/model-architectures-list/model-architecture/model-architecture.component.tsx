// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Divider, Radio, Text } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './model-architecture.module.scss';

interface ModelArchitectureExpandedDescriptionProps {
    description: string;
}

const ModelArchitectureExpandedDescription = ({ description }: ModelArchitectureExpandedDescriptionProps) => {
    return <Text UNSAFE_className={styles.modelArchitectureExpandedDescription}>{description}</Text>;
};

const ModelArchitectureDivider = () => {
    return <Divider size={'S'} />;
};

interface ModelArchitectureParametersProps {
    numberOfParameters: number;
    license?: string;
}

const ModelArchitectureParameters = ({
    numberOfParameters,
    license = 'Apache 2.0',
}: ModelArchitectureParametersProps) => {
    return (
        <ul className={styles.modelArchitectureParameters}>
            <li>Number of parameters: {numberOfParameters} Milions</li>
            <li>License: {license}</li>
        </ul>
    );
};

interface ModelArchitectureName {
    name: string;
    id: string;
    isSelected: boolean;
}

const ModelArchitectureName = ({ name, id, isSelected }: ModelArchitectureName) => {
    return (
        <Radio value={id} UNSAFE_className={styles.modelArchitectureName}>
            {name}
        </Radio>
    );
};

interface ModelArchitectureProps {
    isSelected: boolean;
    children: ReactNode;
    onSelect: () => void;
}

export const ModelArchitecture = ({ isSelected, children, onSelect }: ModelArchitectureProps) => {
    return (
        <div
            className={clsx(styles.modelArchitectureContainer, {
                [styles.modelArchitectureSelected]: isSelected,
            })}
            onClick={onSelect}
        >
            {children}
        </div>
    );
};

ModelArchitecture.Name = ModelArchitectureName;
ModelArchitecture.Parameters = ModelArchitectureParameters;
ModelArchitecture.Divider = ModelArchitectureDivider;
ModelArchitecture.ExpandedDescription = ModelArchitectureExpandedDescription;
