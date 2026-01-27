// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';

export const getAnnotations = (
    mode: AnnotatorMode,
    isUserReviewed: boolean,
    annotations: AnnotationDTO[]
): AnnotationDTO[] => {
    if (mode === 'annotation' && isUserReviewed) {
        return annotations;
    }

    if (mode === 'annotation' && !isUserReviewed) {
        return [];
    }

    if (mode === 'prediction' && isUserReviewed) {
        return [];
    }

    if (mode === 'prediction' && !isUserReviewed) {
        return annotations;
    }

    return [];
};
