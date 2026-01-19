// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Grid, RadioGroup } from '@geti/ui';

import styles from './model-architectures-list-layout.module.scss';

type ModelArchitecturesListLayoutProps = {
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
    ariaLabel: string;
    children: ReactNode;
};

export const ModelArchitecturesListLayout = ({
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
    children,
    ariaLabel,
}: ModelArchitecturesListLayoutProps) => {
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
