// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import { useTrainModelState } from './train-model-provider.component';

export const SelectDatasetRevision = () => {
    const { datasetRevisions, selectedDatasetRevision, onSelectDatasetRevision } = useTrainModelState();

    return (
        <Picker
            flex={1}
            items={datasetRevisions}
            label={'Select dataset revision'}
            selectedKey={selectedDatasetRevision}
            onSelectionChange={(key) => onSelectDatasetRevision(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
