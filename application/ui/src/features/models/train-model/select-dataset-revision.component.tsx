// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import { useTrainModel } from './train-model-provider.component';

export const SelectDatasetRevision = () => {
    const { datasetRevisions, selectedDatasetRevisionId, onSelectDatasetRevisionId } = useTrainModel();

    return (
        <Picker
            flex={1}
            items={datasetRevisions}
            label={'Select dataset revision'}
            selectedKey={selectedDatasetRevisionId}
            onSelectionChange={(key) => onSelectDatasetRevisionId(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
