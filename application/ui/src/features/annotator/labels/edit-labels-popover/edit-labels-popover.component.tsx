// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';

import type { Label } from '../../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../../shared/annotator/labels';
import { EditLabelRow } from '../edit-label-row/edit-label-row.component';
import { NewLabelRow } from '../new-label-row/new-label-row.component';

import classes from './edit-labels-popover.module.scss';

interface EditLabelsPopoverProps {
    labels: Label[];
    onLabelSelect: (label: Label) => void;
    isLabelSelected: (label: Label) => boolean;
    isLabelPinned: (labelId: string) => boolean;
    onRequestDeleteLabel: (label: Label) => void;
    onSaveNewLabel: (name: string, color: string) => void;
    onTogglePinLabel: (label: Label) => void;
    autoCreateNewLabel?: boolean;
}

export const EditLabelsPopover = ({
    labels,
    onLabelSelect,
    isLabelSelected,
    isLabelPinned,
    onRequestDeleteLabel,
    onSaveNewLabel,
    onTogglePinLabel,
    autoCreateNewLabel = false,
}: EditLabelsPopoverProps) => {
    const [isCreatingLabel, setIsCreatingLabel] = useState(autoCreateNewLabel);

    const editableLabels = labels.filter((label) => label.id !== EMPTY_LABEL_ID);

    const handleAddNewLabel = () => {
        setIsCreatingLabel(true);
    };

    const handleSaveNewLabel = (name: string, color: string) => {
        onSaveNewLabel(name, color);
        setIsCreatingLabel(false);
    };

    const handleCancelNewLabel = () => {
        setIsCreatingLabel(false);
    };

    return (
        <View UNSAFE_className={classes.popoverContent}>
            <Flex direction={'column'} gap={'size-50'} UNSAFE_className={classes.labelsList}>
                {editableLabels.map((label) => (
                    <EditLabelRow
                        key={label.id}
                        label={label}
                        existingLabels={editableLabels}
                        isSelected={isLabelSelected(label)}
                        isPinned={isLabelPinned(label.id)}
                        onSelect={() => onLabelSelect(label)}
                        onDelete={onRequestDeleteLabel}
                        onTogglePin={onTogglePinLabel}
                    />
                ))}
                {isCreatingLabel ? (
                    <NewLabelRow
                        existingLabels={editableLabels}
                        onSave={handleSaveNewLabel}
                        onCancel={handleCancelNewLabel}
                    />
                ) : (
                    <ActionButton isQuiet onPress={handleAddNewLabel} aria-label='Add new label'>
                        <Add />
                    </ActionButton>
                )}
            </Flex>
        </View>
    );
};
