// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';

export const getInitialAnnotations = (
    mode: AnnotatorMode,
    isUserReviewed: boolean,
    annotationsDTO: AnnotationDTO[]
): AnnotationDTO[] => {
    return mode === 'annotation' ? (isUserReviewed ? annotationsDTO : []) : [];
};

export const getInitialPredictions = (
    mode: AnnotatorMode,
    isUserReviewed: boolean,
    annotationsDTO: AnnotationDTO[]
): AnnotationDTO[] => {
    return mode === 'prediction' ? (isUserReviewed ? [] : annotationsDTO) : [];
};
