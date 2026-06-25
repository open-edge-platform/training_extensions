// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from 'react';

import type { Annotation as AnnotationType } from '../../../shared/types';

export const AnnotationContext = createContext<AnnotationType | null>(null);

export const useAnnotation = () => {
    const ctx = useContext(AnnotationContext);

    if (ctx === null) {
        throw new Error('useAnnotation must be used within an AnnotationContext.Provider');
    }

    return ctx;
};
