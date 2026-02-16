// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DataValue = {
    stagedDatasetId: string;
};

const LABEL_MAPPING_IMPORT_DATASET_KEY = (projectId: string) => `label-mapping-import-dataset-${projectId}`;

export const useLabelMappingImportDataset = () => {
    const projectId = useProjectIdentifier();

    const [lsLabelMappingImportDataset, setLsLabelMappingImport] = useLocalStorage<DataValue | null>(
        LABEL_MAPPING_IMPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage<DataValue>(LABEL_MAPPING_IMPORT_DATASET_KEY(projectId)) ?? null
    );
    const getLsLabelMappingImport = (): DataValue | null => {
        return lsLabelMappingImportDataset;
    };

    const addLsLabelMappingImport = (stagedDatasetId: string) => {
        return setLsLabelMappingImport({ stagedDatasetId });
    };

    const removeLsLabelMappingImport = (): void => {
        return setLsLabelMappingImport(null);
    };

    return {
        addLsLabelMappingImport,
        getLsLabelMappingImport,
        removeLsLabelMappingImport,
    };
};
