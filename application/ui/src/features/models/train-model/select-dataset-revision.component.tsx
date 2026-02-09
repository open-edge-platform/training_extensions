// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, ContextualHelp, Flex, Heading, Item, Picker } from '@geti/ui';

import { useTrainModel } from './train-model-provider.component';

export const SelectDatasetRevision = () => {
    const { datasetRevisions, selectedDatasetRevisionId, onSelectDatasetRevisionId } = useTrainModel();

    return (
        <Flex alignItems={'center'} gap={'size-100'}>
            <Picker
                flex={1}
                items={datasetRevisions}
                label={'Select dataset'}
                selectedKey={selectedDatasetRevisionId}
                onSelectionChange={(key) => onSelectDatasetRevisionId(String(key))}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            <ContextualHelp variant={'info'} marginTop={'size-300'}>
                <Heading>Selecting a dataset</Heading>
                <Content>
                    Choose the version of the dataset to use for training. If you want to train the new model on the
                    exact same data (media and annotations) as another model, please select the corresponding dataset
                    revision. Conversely, if you want to train on the most recent version of the data (what you see in
                    the &ldquo;Dataset&rdquo; page), please select &ldquo;Use current dataset&rdquo;.
                </Content>
            </ContextualHelp>
        </Flex>
    );
};
