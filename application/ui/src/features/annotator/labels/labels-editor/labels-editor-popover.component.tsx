// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ComponentRef, ReactNode, useRef, useState } from 'react';

import { ActionButton, AlertDialog, DialogContainer, Popover, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { Add, Edit } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { OverlayTriggerState } from 'react-stately';

import type { Label } from '../../../../constants/shared-types';
import { useLabels } from '../use-labels.hook';
import { LabelsEditor } from './labels-editor.component';

type LabelEditorPopoverProps = {
    state: OverlayTriggerState;
    triggerRef: ComponentProps<typeof Popover>['triggerRef'];
    children: ReactNode;
};

const LabelEditorPopover = ({ triggerRef, state, children }: LabelEditorPopoverProps) => {
    return (
        <Popover
            hideArrow
            triggerRef={triggerRef}
            state={state}
            placement={'bottom end'}
            UNSAFE_style={{
                transform: 'translateY(10%)',
            }}
        >
            {children}
        </Popover>
    );
};

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

    const triggerRef = useRef<ComponentRef<'svg'>>(null);
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
                    <ActionButton isQuiet aria-label='Edit labels' onPress={popoverState.open}>
                        <Edit ref={triggerRef} />
                    </ActionButton>
                    <Tooltip>Edit labels</Tooltip>
                </TooltipTrigger>
            ) : (
                <TooltipTrigger>
                    <ActionButton isQuiet aria-label='Create label' onPress={popoverState.open}>
                        <Add ref={triggerRef} />
                        <Text>Create label</Text>
                    </ActionButton>
                    <Tooltip>Create label</Tooltip>
                </TooltipTrigger>
            )}

            <LabelEditorPopover state={popoverState} triggerRef={triggerRef}>
                <LabelsEditor
                    isClassification={isClassification}
                    isMultiLabel={isMultiLabel}
                    onRequestDeleteLabel={handleRequestDeleteLabel}
                    autoCreateNewLabel={!hasLabels}
                />
            </LabelEditorPopover>

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
