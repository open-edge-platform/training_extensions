// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';

export const getInitialAnnotations = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? annotationsDTO : [];
};

export const getInitialPredictions = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? [] : annotationsDTO;
};
