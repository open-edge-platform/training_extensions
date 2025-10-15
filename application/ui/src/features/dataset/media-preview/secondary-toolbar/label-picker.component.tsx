// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Key, Picker } from '@geti/ui';
import { components } from 'src/api/openapi-spec';

type ServerLabel = components['schemas']['Label'];

type LabelPickerProps = {
    labels: ServerLabel[];
    onSelect: (value: Key | null) => void;
    selectedLabel: ServerLabel | null;
};
export const LabelPicker = ({ labels, onSelect, selectedLabel }: LabelPickerProps) => {
    return (
        <Picker selectedKey={selectedLabel?.id} placeholder={'Select label'} onSelectionChange={onSelect}>
            {labels.map((label) => {
                return <Item key={label.id}>{label.name}</Item>;
            })}
        </Picker>
    );
};
