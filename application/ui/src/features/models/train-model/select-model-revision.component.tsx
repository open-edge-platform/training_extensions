// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, ContextualHelp, Heading, Item, Picker } from '@geti/ui';

import { useTrainModelState } from './train-model-provider.component';

export const SelectModelRevision = () => {
    const { modelRevisions, selectedModelRevisionId, onSelectModelRevisionId } = useTrainModelState();

    return (
        <Picker
            flex={1}
            items={modelRevisions}
            label={'Select model revision'}
            selectedKey={selectedModelRevisionId}
            onSelectionChange={(key) => onSelectModelRevisionId(String(key))}
            contextualHelp={
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>Selecting a model revision</Heading>
                    <Content>
                        {`Choose an existing model revision for fine-tuning,
                        or select "Train from scratch" to start from base weights.
                        `}
                    </Content>
                </ContextualHelp>
            }
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
