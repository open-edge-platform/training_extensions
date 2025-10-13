// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Key, Picker } from '@geti/ui';
import { Label } from 'src/features/annotator/types';

type LabelSelectionProps = {
    labels: Label[];
    onSelect: (value: Key | null) => void;
};
export const LabelSelection = ({ labels, onSelect }: LabelSelectionProps) => {
    return (
        <Picker label='Choose a label' onSelectionChange={onSelect}>
            {labels.map((label) => {
                return <Item key={label.id}>{label.name}</Item>;
            })}
        </Picker>
    );
};
