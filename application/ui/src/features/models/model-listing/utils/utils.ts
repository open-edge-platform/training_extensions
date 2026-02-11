// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../../../constants/shared-types';

export const isFailedModel = (model: Pick<Model, 'training_info'>): boolean => model.training_info?.status === 'failed';

export const isTrainingModel = (model: Pick<Model, 'training_info'>): boolean =>
    model.training_info?.status === 'in_progress';
