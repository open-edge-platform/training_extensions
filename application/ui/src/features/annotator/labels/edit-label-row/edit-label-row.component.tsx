// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, useState } from 'react';

import { ActionButton, Checkbox, Flex, TextField, View } from '@geti/ui';
import { Delete } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { HotkeyField } from '../../../../components/label-fields/hotkey-field.component';
import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { validateLabelHotkey, validateLabelName } from '../../../../components/label-fields/label-validation';
import type { Label } from '../../../../constants/shared-types';
import { useUpdateLabel } from '../api/use-update-label.hook';
import { DeleteLabelDialog } from '../delete-label-dialog/delete-label-dialog.component';

import classes from './edit-label-row.module.scss';

interface EditLabelRowProps {
    label: Label;
    existingLabels: Label[];
    allHotkeys: string[];
    isSelected: boolean;
    onSelect: () => void;
    onDelete: (labelId: string) => void;
}

export const EditLabelRow = ({
    label,
    existingLabels,
    allHotkeys,
    isSelected,
    onSelect,
    onDelete,
}: EditLabelRowProps) => {
    const projectId = useProjectIdentifier();
    const updateLabelMutation = useUpdateLabel();

    const [name, setName] = useState(label.name);
    const [color, setColor] = useState(label.color);
    const [hotkey, setHotkey] = useState<string | null>(label.hotkey ?? null);

    const validationError = validateLabelName(name, existingLabels, label.id);

    const getValidName = () => (validationError || name.trim() === '' ? label.name : name.trim());

    const handleNameBlur = () => {
        if (validationError || name.trim() === '') return;
        if (name === label.name) return;

        updateLabelMutation.mutate({
            body: {
                labels_to_edit: [{ id: label.id, new_name: name.trim(), new_color: color, new_hotkey: hotkey }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const handleColorChange = (newColor: string) => {
        setColor(newColor);

        updateLabelMutation.mutate({
            body: {
                labels_to_edit: [{ id: label.id, new_name: getValidName(), new_color: newColor, new_hotkey: hotkey }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const handleHotkeyChange = (newHotkey: string | null) => {
        setHotkey(newHotkey);

        if (newHotkey !== null && validateLabelHotkey(newHotkey, allHotkeys)) {
            return;
        }

        updateLabelMutation.mutate({
            body: {
                labels_to_edit: [{ id: label.id, new_name: getValidName(), new_color: color, new_hotkey: newHotkey }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const handleDelete = () => {
        onDelete(label.id);
    };

    return (
        <Flex
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color } as CSSProperties}
        >
            <Checkbox isSelected={isSelected} onChange={() => onSelect()} aria-label={`Select ${label.name} label`} />

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

            <View width={'size-1600'}>
                <HotkeyField hotkey={hotkey} onHotkeyChange={handleHotkeyChange} allHotkeys={allHotkeys} />
            </View>

            <DeleteLabelDialog label={label} onDelete={handleDelete}>
                <ActionButton aria-label={`Delete ${label.name} label`} isQuiet UNSAFE_className={classes.deleteButton}>
                    <Delete />
                </ActionButton>
            </DeleteLabelDialog>
        </Flex>
    );
};
