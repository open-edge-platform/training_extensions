// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, ContextualHelp, Heading, Item, Picker } from '@geti-ui/ui';

import { useTrainModelState } from './train-model-provider.component';

export const SelectDatasetRevision = () => {
    const { datasetRevisions, selectedDatasetRevisionId, onSelectDatasetRevisionId } = useTrainModelState();

    return (
        <>
            <Picker
                flex={1}
                items={datasetRevisions}
                label={'Select dataset'}
                selectedKey={selectedDatasetRevisionId}
                onSelectionChange={(key) => onSelectDatasetRevisionId(String(key))}
                contextualHelp={
                    <ContextualHelp variant={'info'} placement={'top'}>
                        <Heading>Selecting a dataset</Heading>
                        <Content>
                            {`Choose the version of the dataset to use for training. If you want to train the new model
                            on the exact same data (media and annotations) as another model, please select the
                            corresponding dataset revision. Conversely, if you want to train on the most recent version
                            of the data (what you see in the "Dataset" page), please select "Use
                            current dataset".`}
                        </Content>
                    </ContextualHelp>
                }
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>
        </>
    );
};
