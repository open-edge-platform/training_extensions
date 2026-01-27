// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';

export const getAnnotations = (
    mode: AnnotatorMode,
    isUserReviewed: boolean,
    annotations: AnnotationDTO[]
): AnnotationDTO[] => {
    if (mode === 'annotation') {
        return isUserReviewed ? annotations : [];
    }
    if (mode === 'prediction') {
        return isUserReviewed ? [] : annotations;
    }

    return [];
};
