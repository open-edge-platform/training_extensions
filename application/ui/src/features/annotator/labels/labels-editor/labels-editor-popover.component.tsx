// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    CustomPopover,
    DialogContainer,
    FocusableRefValue,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti/ui';
import { Add, Edit } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';

import type { Label } from '../../../../constants/shared-types';
import { useLabels } from '../use-labels.hook';
import { LabelsEditor } from './labels-editor.component';

type LabelsEditorPopoverProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
    hasLabels: boolean;
};

export const LabelsEditorPopover = ({
    isClassification = false,
    isMultiLabel = false,
    hasLabels,
}: LabelsEditorPopoverProps) => {
    const { deleteLabel } = useLabels({ isClassification, isMultiLabel });

    const triggerRef = useRef<FocusableRefValue<HTMLElement, HTMLButtonElement>>(null);
    const popoverState = useOverlayTriggerState({});
    const deleteDialogState = useOverlayTriggerState({});
    const [labelToDelete, setLabelToDelete] = useState<Label | null>(null);

    const handleRequestDeleteLabel = (label: Label) => {
        setLabelToDelete(label);
        popoverState.close();
        deleteDialogState.open();
    };

    const handleConfirmDeleteLabel = () => {
        if (labelToDelete) {
            deleteDialogState.close();
            deleteLabel(labelToDelete.id);
            setLabelToDelete(null);
        }
    };

    const handleCancelDeleteLabel = () => {
        deleteDialogState.close();
        setLabelToDelete(null);
        popoverState.open();
    };

    return (
        <>
            {hasLabels ? (
                <TooltipTrigger>
                    <ActionButton ref={triggerRef} isQuiet aria-label='Edit labels' onPress={popoverState.open}>
                        <Edit />
                    </ActionButton>
                    <Tooltip>Edit labels</Tooltip>
                </TooltipTrigger>
            ) : (
                <TooltipTrigger>
                    <ActionButton ref={triggerRef} isQuiet aria-label='Create label' onPress={popoverState.open}>
                        <Add />
                        <Text>Create label</Text>
                    </ActionButton>
                    <Tooltip>Create label</Tooltip>
                </TooltipTrigger>
            )}
            {popoverState.isOpen && (
                <CustomPopover ref={triggerRef} state={popoverState} placement='bottom end'>
                    <LabelsEditor
                        isClassification={isClassification}
                        isMultiLabel={isMultiLabel}
                        onRequestDeleteLabel={handleRequestDeleteLabel}
                        autoCreateNewLabel={!hasLabels}
                    />
                </CustomPopover>
            )}

            <DialogContainer onDismiss={handleCancelDeleteLabel}>
                {deleteDialogState.isOpen && labelToDelete && (
                    <AlertDialog
                        title={'Delete label'}
                        variant={'destructive'}
                        primaryActionLabel={'Delete'}
                        cancelLabel={'Cancel'}
                        onPrimaryAction={handleConfirmDeleteLabel}
                        onCancel={handleCancelDeleteLabel}
                    >
                        If you remove the {labelToDelete.name} label, all annotations in your dataset that have this
                        label will be deleted. However, this won&apos;t impact any models you&apos;ve trained in the
                        past.
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
