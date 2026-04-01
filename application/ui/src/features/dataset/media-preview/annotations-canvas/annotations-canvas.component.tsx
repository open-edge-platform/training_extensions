// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti/ui';

import classes from './annotations-canvas.module.scss';

type AnnotationsCanvasProps = {
    children: ReactNode;
    isReadOnly?: boolean;
};

export const AnnotationsCanvas = ({ children, isReadOnly = false }: AnnotationsCanvasProps) => {
    return (
        <View
            gridArea={'canvas'}
            overflow={'hidden'}
            UNSAFE_className={isReadOnly ? classes.readOnlyCanvas : undefined}
        >
            {children}
        </View>
    );
};
