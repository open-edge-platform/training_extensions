// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../../../constants/shared-types';

const TRAINING_STATUS = {
    Failed: 'failed',
    InProgress: 'in_progress',
    Successful: 'successful',
} as const;

export const isFailedModel = (model: Pick<Model, 'training_info'>): boolean =>
    model.training_info?.status === TRAINING_STATUS.Failed;

export const isTrainingModel = (model: Pick<Model, 'training_info'>): boolean =>
    model.training_info?.status === TRAINING_STATUS.InProgress;

export const isSuccessfulModel = (model: Pick<Model, 'training_info'>): boolean =>
    model.training_info?.status === TRAINING_STATUS.Successful;

export const hasDeletedWeights = (model: Pick<Model, 'files_deleted'>): boolean => model.files_deleted;
