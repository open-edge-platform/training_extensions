// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, useState } from 'react';

import { ActionButton, Flex, TextField, View } from '@geti/ui';
import { Delete } from '@geti/ui/icons';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { SilentCheckbox } from '../../../../components/label-fields/silent-checkbox.component';
import type { Label } from '../../../../constants/shared-types';

import classes from './label-row.module.scss';

type LabelRowProps = {
    label: Label;
    isSelected: boolean;
    onSelect: () => void;
    onDelete: (label: Label) => void;
    onUpdate: (labelId: string, updates: { name: string; color: string; hotkey: string | null | undefined }) => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
};

export const LabelRow = ({ label, isSelected, onSelect, onDelete, onUpdate, validateName }: LabelRowProps) => {
    const [name, setName] = useState(label.name);
    const [color, setColor] = useState(label.color);

    const validationError = validateName(name, label.id);

    const getValidName = () => (validationError || name.trim() === '' ? label.name : name.trim());

    const handleNameBlur = () => {
        if (validationError || name.trim() === '') return;
        if (name === label.name) return;

        onUpdate(label.id, { name: name.trim(), color, hotkey: label.hotkey });
    };

    const handleColorChange = (newColor: string) => {
        setColor(newColor);
        onUpdate(label.id, { name: getValidName(), color: newColor, hotkey: label.hotkey });
    };

    return (
        <Flex
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color } as CSSProperties}
        >
            <SilentCheckbox isSelected={isSelected} onChange={onSelect} aria-label={`Select ${label.name} label`} />

            <LabelColorPicker color={color} onChange={handleColorChange} />

            <View flex={1}>
                <TextField
                    aria-label={'Label name'}
                    placeholder={'Label name'}
                    value={name}
                    onChange={setName}
                    onBlur={handleNameBlur}
                    width={'100%'}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                />
            </View>

            <ActionButton
                aria-label={`Delete ${label.name} label`}
                isQuiet
                UNSAFE_className={classes.deleteButton}
                onPress={() => onDelete(label)}
            >
                <Delete />
            </ActionButton>
        </Flex>
    );
};
