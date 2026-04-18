// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ImportDatasetToProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { useProjectIdentifier } from '../use-project-identifier.hook';
import { useDatasetImportStorage } from './use-dataset-import-storage.hook';
import { DatasetImportState } from './utils';

export const IMPORT_DATASET_TO_PROJECT_PREFIX = 'import-dataset-to-project-';

const IMPORT_DATASET_TO_PROJECT_KEY = (projectId: string) => `${IMPORT_DATASET_TO_PROJECT_PREFIX}${projectId}`;

export const useImportDatasetToProject = () => {
    const projectId = useProjectIdentifier();

    return useDatasetImportStorage<ImportDatasetToProjectState, DatasetImportState<ImportDatasetToProjectState>>(
        IMPORT_DATASET_TO_PROJECT_KEY(projectId)
    );
};
