// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { AnnotationShape } from './annotation-shape.component';
import { useAnnotation } from './annotation.component';

type EditAnnotationProps = {
    children: ReactNode;
};

export const EditAnnotation = ({ children }: EditAnnotationProps) => {
    const selectedAnnotations = useSelectedAnnotations();
    const annotation = useAnnotation();

    if (selectedAnnotations.has(annotation.id)) {
        return <AnnotationShape annotation={annotation} />;
    }

    return <>{children}</>;
};
