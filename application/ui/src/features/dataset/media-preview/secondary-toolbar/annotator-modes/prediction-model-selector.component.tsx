// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

type PredictionModelSelectorProps = {
    isDisabled: boolean;
};

export const PredictionModelSelector = ({ isDisabled }: PredictionModelSelectorProps) => {
    const { selectableModels, selectedModelId, changeSelectedModelId } = usePredictionSetup();

    if (isEmpty(selectableModels)) {
        return null;
    }

    return (
        <Picker
            isQuiet
            aria-label={'Select prediction model'}
            items={selectableModels}
            selectedKey={selectedModelId}
            isDisabled={isDisabled}
            onSelectionChange={(key) => key !== null && changeSelectedModelId(String(key))}
        >
            {(item) => <Item key={item.modelVariantId}>{item.name}</Item>}
        </Picker>
    );
};
