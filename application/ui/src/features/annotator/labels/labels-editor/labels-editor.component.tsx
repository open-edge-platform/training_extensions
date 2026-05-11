// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Loading, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Label } from '../../../../constants/shared-types';
import { usePinnedLabels } from '../hooks/use-pinned-labels.hook';
import { LabelRow } from '../label-row/label-row.component';
import { NewLabelRow } from '../new-label-row/new-label-row.component';
import { useLabels } from '../use-labels.hook';

import classes from './labels-editor.module.scss';

type LabelsEditorProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
    onRequestDeleteLabel: (label: Label) => void;
    autoCreateNewLabel?: boolean;
};

export const LabelsEditor = ({
    isClassification = false,
    isMultiLabel = false,
    onRequestDeleteLabel,
    autoCreateNewLabel = false,
}: LabelsEditorProps) => {
    const { editableLabels, isUpdating, toggleLabelOnAnnotations, isLabelActive, addLabel, updateLabel, validateName } =
        useLabels({
            isClassification,
            isMultiLabel,
        });

    const projectId = useProjectIdentifier();
    const { isPinned, togglePin } = usePinnedLabels(projectId);

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

    const handleTogglePin = (label: Label) => {
        togglePin(label.id);
    };

    return (
        <View UNSAFE_className={classes.editorContent}>
            <Flex direction={'column'} gap={'size-50'} UNSAFE_className={classes.labelsList}>
                {editableLabels.map((label) => (
                    <LabelRow
                        key={label.id}
                        label={label}
                        isSelected={isLabelActive(label)}
                        isPinned={isPinned(label.id)}
                        onSelect={() => toggleLabelOnAnnotations(label)}
                        onDelete={onRequestDeleteLabel}
                        onTogglePin={handleTogglePin}
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
                        {isUpdating ? <Loading mode='inline' size='S' /> : <Add />}
                    </ActionButton>
                )}
            </Flex>
        </View>
    );
};
