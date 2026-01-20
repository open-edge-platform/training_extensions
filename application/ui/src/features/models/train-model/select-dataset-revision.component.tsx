// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import { DatasetRevision } from '../../../constants/shared-types';

type SelectDatasetRevisionProps = {
    datasetRevisions: DatasetRevision[];
    selectedDatasetRevision: string | null;
    onSelectedDatasetRevisionChange: (datasetRevision: string) => void;
};

export const SelectDatasetRevision = ({
    datasetRevisions,
    selectedDatasetRevision,
    onSelectedDatasetRevisionChange,
}: SelectDatasetRevisionProps) => {
    return (
        <Picker
            flex={1}
            items={datasetRevisions}
            label={'Select dataset revision'}
            selectedKey={selectedDatasetRevision}
            onSelectionChange={(key) => onSelectedDatasetRevisionChange(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
