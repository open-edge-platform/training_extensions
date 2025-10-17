// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Key, Picker } from '@geti/ui';
import { Label } from 'src/features/annotator/types';

type LabelPickerProps = {
    labels: Label[];
    onSelect: (value: Key | null) => void;
    selectedLabel: Label | null;
};
export const LabelPicker = ({ labels, onSelect, selectedLabel }: LabelPickerProps) => {
    return (
        <Picker
            selectedKey={selectedLabel?.id}
            placeholder={'Select label'}
            onSelectionChange={onSelect}
            aria-label='Label Picker'
        >
            {labels.map((label) => {
                return <Item key={label.id}>{label.name}</Item>;
            })}
        </Picker>
    );
};
