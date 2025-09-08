// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useSelectedAnnotation } from '../select-annotation-provider.component';
import { useAnnotation } from './annotation.component';

type EditAnnotationProps = {
    children: ReactNode;
};

export const EditAnnotation = ({ children }: EditAnnotationProps) => {
    const selectedAnnotations = useSelectedAnnotation();
    const annotation = useAnnotation();

    if (selectedAnnotations.has(annotation)) {
        return null;
    }

    return <>{children}</>;
};
