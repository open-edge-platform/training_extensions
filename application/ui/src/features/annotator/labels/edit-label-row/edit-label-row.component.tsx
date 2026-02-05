// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, useState } from 'react';

import { ActionButton, Flex, TextField, View } from '@geti/ui';
import { Delete } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { validateLabelName } from '../../../../components/label-fields/label-validation';
import { SilentCheckbox } from '../../../../components/label-fields/silent-checkbox.component';
import type { Label } from '../../../../constants/shared-types';
import { useUpdateLabel } from '../api/use-update-label.hook';

import classes from './edit-label-row.module.scss';

interface EditLabelRowProps {
    label: Label;
    existingLabels: Label[];
    isSelected: boolean;
    onSelect: () => void;
    onDelete: (label: Label) => void;
}

export const EditLabelRow = ({ label, existingLabels, isSelected, onSelect, onDelete }: EditLabelRowProps) => {
    const projectId = useProjectIdentifier();
    const updateLabelMutation = useUpdateLabel();

    const [name, setName] = useState(label.name);
    const [color, setColor] = useState(label.color);

    const validationError = validateLabelName(name, existingLabels, label.id);

    const getValidName = () => (validationError || name.trim() === '' ? label.name : name.trim());

    const handleNameBlur = () => {
        if (validationError || name.trim() === '') return;
        if (name === label.name) return;

        updateLabelMutation.mutate({
            body: {
                labels_to_edit: [{ id: label.id, new_name: name.trim(), new_color: color, new_hotkey: label.hotkey }],
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
                labels_to_edit: [
                    { id: label.id, new_name: getValidName(), new_color: newColor, new_hotkey: label.hotkey },
                ],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    return (
        <Flex
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color } as CSSProperties}
        >
            <SilentCheckbox
                isSelected={isSelected}
                onChange={onSelect}
                aria-label={`Select ${label.name} label`}
            />

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
