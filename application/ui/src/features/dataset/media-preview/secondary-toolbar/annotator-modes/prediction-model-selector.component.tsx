// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

type PredictionModelSelectorProps = {
    isDisabled: boolean;
};

export const PredictionModelSelector = ({ isDisabled }: PredictionModelSelectorProps) => {
    const { models, selectedModelId, changeSelectedModelId } = usePredictionSetup();

    if (isEmpty(models)) {
        return null;
    }

    return (
        <Picker
            isQuiet
            aria-label={'Select prediction model'}
            items={models}
            selectedKey={selectedModelId}
            isDisabled={isDisabled}
            onSelectionChange={(key) => key !== null && changeSelectedModelId(String(key))}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
