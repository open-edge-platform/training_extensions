// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';

import type { Label } from '../../../../constants/shared-types';
import { LabelRow } from '../label-row/label-row.component';
import { NewLabelRow } from '../new-label-row/new-label-row.component';
import { useLabels } from '../use-labels.hook';

import classes from './labels-popover.module.scss';

type LabelsPopoverProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
    onRequestDeleteLabel: (label: Label) => void;
    autoCreateNewLabel?: boolean;
};

export const LabelsPopover = ({
    isClassification = false,
    isMultiLabel = false,
    onRequestDeleteLabel,
    autoCreateNewLabel = false,
}: LabelsPopoverProps) => {
    const { editableLabels, toggleLabelOnAnnotations, isLabelActive, addLabel, updateLabel, validateName } = useLabels({
        isClassification,
        isMultiLabel,
    });

    const [isCreatingLabel, setIsCreatingLabel] = useState(autoCreateNewLabel);

    const handleAddNewLabel = () => {
        setIsCreatingLabel(true);
    };

    const handleSaveNewLabel = (name: string, color: string) => {
        addLabel(name, color);
        setIsCreatingLabel(false);
    };

    const handleCancelNewLabel = () => {
        setIsCreatingLabel(false);
    };

    return (
        <View UNSAFE_className={classes.popoverContent}>
            <Flex direction={'column'} gap={'size-50'} UNSAFE_className={classes.labelsList}>
                {editableLabels.map((label) => (
                    <LabelRow
                        key={label.id}
                        label={label}
                        isSelected={isLabelActive(label)}
                        onSelect={() => toggleLabelOnAnnotations(label)}
                        onDelete={onRequestDeleteLabel}
                        onUpdate={updateLabel}
                        validateName={validateName}
                    />
                ))}
                {isCreatingLabel ? (
                    <NewLabelRow
                        onSave={handleSaveNewLabel}
                        onCancel={handleCancelNewLabel}
                        validateName={validateName}
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
