// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Label, TaskType } from '../../../../constants/shared-types';
import { TASK_HOTKEYS } from '../../../../shared/hotkeys-definition';
import { useUpdateLabel } from '../api/use-update-label.hook';
import { EditLabelRow } from '../edit-label-row/edit-label-row.component';
import { NewLabelRow } from '../new-label-row/new-label-row.component';

import classes from './edit-labels-popover.module.scss';

interface EditLabelsPopoverProps {
    labels: Label[];
    taskType: TaskType;
    onLabelSelect: (label: Label) => void;
    isLabelSelected: (label: Label) => boolean;
    autoCreateNewLabel?: boolean;
}

export const EditLabelsPopover = ({
    labels,
    taskType,
    onLabelSelect,
    isLabelSelected,
    autoCreateNewLabel = false,
}: EditLabelsPopoverProps) => {
    const projectId = useProjectIdentifier();
    const updateLabelMutation = useUpdateLabel();
    const [isCreatingLabel, setIsCreatingLabel] = useState(autoCreateNewLabel);

    // Collect all hotkeys: existing labels + app hotkeys
    const labelsHotkeys = labels.map((label) => label.hotkey).filter((hotkey): hotkey is string => hotkey != null);
    const appHotkeys = Object.values(TASK_HOTKEYS[taskType]);
    const allHotkeys = [...labelsHotkeys, ...appHotkeys];

    const handleDeleteLabel = (labelId: string) => {
        updateLabelMutation.mutate({
            body: {
                labels_to_remove: [{ id: labelId }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const handleAddNewLabel = () => {
        setIsCreatingLabel(true);
    };

    const handleSaveNewLabel = (name: string, color: string, hotkey: string | null) => {
        updateLabelMutation.mutate({
            body: {
                labels_to_add: [{ name, color, hotkey }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
        setIsCreatingLabel(false);
    };

    const handleCancelNewLabel = () => {
        setIsCreatingLabel(false);
    };

    return (
        <View UNSAFE_className={classes.popoverContent}>
            <Flex direction={'column'} gap={'size-50'} UNSAFE_className={classes.labelsList}>
                {labels.map((label) => (
                    <EditLabelRow
                        key={label.id}
                        label={label}
                        existingLabels={labels}
                        allHotkeys={allHotkeys}
                        isSelected={isLabelSelected(label)}
                        onSelect={() => onLabelSelect(label)}
                        onDelete={handleDeleteLabel}
                    />
                ))}
                {isCreatingLabel ? (
                    <NewLabelRow
                        existingLabels={labels}
                        allHotkeys={allHotkeys}
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
