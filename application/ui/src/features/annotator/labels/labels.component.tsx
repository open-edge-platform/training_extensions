// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, Fragment, useRef, useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    CustomPopover,
    DialogContainer,
    Divider,
    Flex,
    FocusableRefValue,
    Text,
} from '@geti/ui';
import { Add, Edit } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { clsx } from 'clsx';

import type { Label } from '../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { LabelsPopover } from './labels-popover/labels-popover.component';
import { useLabels } from './use-labels.hook';

import classes from './labels.module.scss';

type LabelBadgeProps = {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
};

const LabelBadge = ({ label, isSelected, onClick }: LabelBadgeProps) => {
    return (
        <button
            onClick={onClick}
            style={{ '--labelBgColor': label.color } as CSSProperties}
            className={clsx(classes.badge, { [classes.selected]: isSelected })}
            aria-pressed={isSelected}
            aria-label={`Label ${label.name}`}
        >
            <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
        </button>
    );
};

type LabelsProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
};

export const Labels = ({ isClassification = false, isMultiLabel = false }: LabelsProps) => {
    const { labels, hasLabels, toggleLabelOnAnnotations, isLabelActive, deleteLabel } = useLabels({
        isClassification,
        isMultiLabel,
    });

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
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            {hasLabels && (
                <div aria-label={'Labels'} className={classes.labelsContainer}>
                    {labels.map((label) => (
                        <Fragment key={label.id}>
                            {label.id === EMPTY_LABEL_ID && <Divider size={'S'} orientation={'vertical'} />}
                            <LabelBadge
                                label={label}
                                isSelected={isLabelActive(label)}
                                onClick={() => toggleLabelOnAnnotations(label)}
                            />
                        </Fragment>
                    ))}
                </div>
            )}
            {hasLabels ? (
                <ActionButton ref={triggerRef} isQuiet aria-label='Edit labels' onPress={popoverState.open}>
                    <Edit />
                </ActionButton>
            ) : (
                <ActionButton ref={triggerRef} isQuiet aria-label='Create label' onPress={popoverState.open}>
                    <Add />
                    <Text>Create label</Text>
                </ActionButton>
            )}
            {popoverState.isOpen && (
                <CustomPopover ref={triggerRef} state={popoverState} placement='bottom end'>
                    <LabelsPopover
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
        </Flex>
    );
};
