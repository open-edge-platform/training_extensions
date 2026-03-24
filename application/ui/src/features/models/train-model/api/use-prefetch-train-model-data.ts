// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchPipeline } from 'hooks/api/pipeline.hook';
import { usePrefetchDatasetItems } from 'hooks/use-get-dataset-items.hook';
import { usePrefetchDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';

import { usePrefetchTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { usePrefetchModels } from '../../hooks/api/use-get-models.hook';
import { REVIEWED_DATASET_ITEMS_OPTIONS } from '../hooks/use-is-training-button-disabled';
import { usePrefetchTrainingDevices } from './use-get-training-devices';

export const usePrefetchTrainModelData = () => {
    usePrefetchTaskModelArchitectures();
    usePrefetchTrainingDevices();
    usePrefetchDatasetRevisions();
    usePrefetchModels();
    usePrefetchPipeline();

    usePrefetchDatasetItems(REVIEWED_DATASET_ITEMS_OPTIONS);
};
