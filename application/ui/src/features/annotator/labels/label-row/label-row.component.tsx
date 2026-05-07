// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent, useState } from 'react';

import { ActionButton, Grid, TextField, Tooltip, TooltipTrigger, View } from '@geti/ui';
import { Delete, Pin, Unpin } from '@geti/ui/icons';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { SilentCheckbox } from '../../../../components/label-fields/silent-checkbox.component';
import type { Label } from '../../../../constants/shared-types';
import { useDebounce } from '../../../../hooks/use-debounce.hook';

import classes from './label-row.module.scss';

const COLOR_DEBOUNCE_MS = 300;

export type LabelRowProps = {
    label: Label;
    isSelected: boolean;
    isPinned: boolean;
    onSelect: () => void;
    onDelete: (label: Label) => void;
    onTogglePin: (label: Label) => void;
    onUpdate: (labelId: string, updates: { name: string; color: string; hotkey: string | null | undefined }) => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
};

const onEnter = (handler: () => void) => (event: KeyboardEvent) => {
    event.key === 'Enter' && handler();
};

export const LabelRow = ({
    label,
    isSelected,
    isPinned,
    onSelect,
    onDelete,
    onTogglePin,
    onUpdate,
    validateName,
}: LabelRowProps) => {
    const [name, setName] = useState(label.name);
    const [color, setColor] = useState(label.color);

    const validationError = validateName(name, label.id);

    const getValidName = () => (validationError || name.trim() === '' ? label.name : name.trim());

    const handleUpdateName = () => {
        if (validationError || name.trim() === '') return;
        if (name === label.name) return;

        onUpdate(label.id, { name: name.trim(), color, hotkey: label.hotkey });
    };

    const debouncedUpdate = useDebounce(
        (newColor: string, currentName: string) => {
            onUpdate(label.id, { name: currentName, color: newColor, hotkey: label.hotkey });
        },
        COLOR_DEBOUNCE_MS,
        [onUpdate, label.id, label.hotkey]
    );

    const handleColorChange = (newColor: string) => {
        setColor(newColor);
        debouncedUpdate(newColor, getValidName());
    };

    return (
        <Grid
            columns={['size-350', 'size-400', '1fr', 'size-400', 'size-400']}
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color }}
        >
            <SilentCheckbox isSelected={isSelected} onChange={onSelect} aria-label={`Select ${label.name} label`} />

            <LabelColorPicker color={color} onChange={handleColorChange} />

            <View>
                <TextField
                    width={'100%'}
                    value={name}
                    aria-label={'Label name'}
                    placeholder={'Label name'}
                    onChange={setName}
                    onBlur={handleUpdateName}
                    onKeyDown={onEnter(handleUpdateName)}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                />
            </View>

            <TooltipTrigger>
                <ActionButton
                    aria-label={isPinned ? `Unpin ${label.name} label` : `Pin ${label.name} label`}
                    isQuiet
                    UNSAFE_className={isPinned ? classes.pinButtonPinned : classes.pinButton}
                    onPress={() => onTogglePin(label)}
                >
                    {isPinned ? <Pin /> : <Unpin />}
                </ActionButton>
                <Tooltip>{isPinned ? 'Unpin label' : 'Pin label'}</Tooltip>
            </TooltipTrigger>

            <ActionButton
                aria-label={`Delete ${label.name} label`}
                isQuiet
                UNSAFE_className={classes.deleteButton}
                onPress={() => onDelete(label)}
            >
                <Delete />
            </ActionButton>
        </Grid>
    );
};
