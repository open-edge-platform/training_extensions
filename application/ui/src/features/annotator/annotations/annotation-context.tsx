// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from 'react';

import { Annotation as AnnotationType } from '../types';

const AnnotationContext = createContext<AnnotationType | null>(null);

export const useAnnotation = () => {
    const ctx = useContext(AnnotationContext);

    return ctx!;
};

export { AnnotationContext };
