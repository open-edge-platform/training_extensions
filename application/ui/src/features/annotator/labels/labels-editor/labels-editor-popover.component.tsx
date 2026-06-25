// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, AlertDialog, DialogContainer, DialogTrigger, Text, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Add, Edit } from '@geti-ui/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';

import type { Label } from '../../../../constants/shared-types';
import { useLabels } from '../use-labels.hook';
import { LabelsEditor } from './labels-editor.component';

const POPOVER_OFFSET_ALIGNMENT = 8;

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

    const triggerLabel = hasLabels ? 'Edit labels' : 'Create label';

    return (
        <>
            <DialogTrigger
                type='popover'
                hideArrow
                isOpen={popoverState.isOpen}
                onOpenChange={popoverState.setOpen}
                placement='bottom end'
                offset={POPOVER_OFFSET_ALIGNMENT}
                crossOffset={POPOVER_OFFSET_ALIGNMENT}
            >
                <TooltipTrigger>
                    <ActionButton isQuiet aria-label={triggerLabel}>
                        {hasLabels ? (
                            <Edit />
                        ) : (
                            <>
                                <Add />
                                <Text>Create label</Text>
                            </>
                        )}
                    </ActionButton>
                    <Tooltip>{triggerLabel}</Tooltip>
                </TooltipTrigger>

                <LabelsEditor
                    isClassification={isClassification}
                    isMultiLabel={isMultiLabel}
                    onRequestDeleteLabel={handleRequestDeleteLabel}
                    autoCreateNewLabel={!hasLabels}
                />
            </DialogTrigger>

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
