// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useAnnotator } from '../annotator-provider.component';
import { useAnnotation } from './annotation.component';

type EditAnnotationProps = {
    children: ReactNode;
};

export const EditAnnotation = ({ children }: EditAnnotationProps) => {
    const { selectedAnnotation } = useAnnotator();
    const annotation = useAnnotation();

    if (selectedAnnotation?.id === annotation.id) return null;

    return <>{children}</>;
};
