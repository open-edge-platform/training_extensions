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
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation } from '../../../shared/types';
import { toggleLabel } from '../../dataset/media-preview/secondary-toolbar/util';
import { useUpdateLabel } from './api/use-update-label.hook';
import { EditLabelsPopover } from './edit-labels-popover/edit-labels-popover.component';

import classes from './labels.module.scss';

interface LabelBadgeProps {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
}

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

interface LabelsProps {
    isClassification?: boolean;
    isMultiLabel?: boolean;
}

const filterOutEmptyLabels = (labels: Label[]): Label[] => labels.filter((label) => label.id !== EMPTY_LABEL_ID);

export const Labels = ({ isClassification = false, isMultiLabel = false }: LabelsProps) => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, addAnnotations, updateAnnotations, deleteAnnotations, addAnnotationWithEmptyLabel } =
        useAnnotationActions();

    const projectId = useProjectIdentifier();
    const updateLabelMutation = useUpdateLabel();

    const triggerRef = useRef<FocusableRefValue<HTMLElement, HTMLButtonElement>>(null);
    const popoverState = useOverlayTriggerState({});
    const deleteDialogState = useOverlayTriggerState({});
    const [labelToDelete, setLabelToDelete] = useState<Label | null>(null);

    const handleClassificationClick = (label: Label) => {
        if (isEmpty(annotations)) {
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (label.id === EMPTY_LABEL_ID && annotations.length !== 0) {
            addAnnotationWithEmptyLabel(label);
            return;
        }

        if (isMultiLabel) {
            const hasEmptyLabel = annotations.some((annotation) =>
                annotation.labels.some((l) => l.id === EMPTY_LABEL_ID)
            );

            let annotationsToUpdate = annotations;

            // If there is an annotation with the empty label, and we are trying to add assign new label to it, we
            // need to remove the empty label from the annotation first.
            if (hasEmptyLabel) {
                annotationsToUpdate = annotations.map((annotation) => ({
                    ...annotation,
                    labels: filterOutEmptyLabels(annotation.labels),
                }));
            }

            const updatedAnnotations = annotationsToUpdate.map((annotation) => ({
                ...annotation,
                labels: toggleLabel(label, annotation.labels),
            }));

            const hasNoLabels = updatedAnnotations.every(({ labels: annotationLabels }) => isEmpty(annotationLabels));

            if (hasNoLabels) {
                deleteAnnotations(updatedAnnotations.map(({ id }) => id));
            } else {
                updateAnnotations(updatedAnnotations);
            }
        } else {
            const isAlreadySelected = annotations.some((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (isAlreadySelected) {
                deleteAnnotations(annotations.map(({ id }) => id));
            } else {
                updateAnnotations(annotations.map((annotation) => ({ ...annotation, labels: [label] })));
            }
        }
    };

    const handleNonClassificationClick = (label: Label) => {
        if (label.id === EMPTY_LABEL_ID) {
            addAnnotationWithEmptyLabel(label);
            return;
        }

        if (selectedAnnotations.size > 0) {
            const selectedAnnotationsList = annotations.filter((a) => selectedAnnotations.has(a.id));

            const allAnnotationsHaveLabel = selectedAnnotationsList.every((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (allAnnotationsHaveLabel) {
                // Remove label
                const updatedAnnotations = selectedAnnotationsList.map((annotation) => {
                    const filteredLabels = annotation.labels.filter((l) => l.id !== label.id);
                    return { ...annotation, labels: filteredLabels } as Annotation;
                });
                updateAnnotations(updatedAnnotations);
                setSelectedLabelId(null);
            } else {
                // Add label
                updateAnnotations(selectedAnnotationsList, [label]);
                setSelectedLabelId(label.id);
            }
        } else {
            setSelectedLabelId(label.id);
        }
    };

    const handleLabelClick = (label: Label) => {
        if (isClassification) {
            handleClassificationClick(label);
        } else {
            handleNonClassificationClick(label);
        }
    };

    const isLabelSelected = (label: Label): boolean => {
        if (label.id === EMPTY_LABEL_ID) {
            return false;
        }

        if (isClassification) {
            return annotations.some((annotation) => annotation.labels.some((l) => l.id === label.id));
        }
        return selectedLabelId === label.id;
    };

    const handleRequestDeleteLabel = (label: Label) => {
        setLabelToDelete(label);
        popoverState.close();
        deleteDialogState.open();
    };

    const handleConfirmDeleteLabel = () => {
        if (labelToDelete) {
            deleteDialogState.close();
            updateLabelMutation.mutate({
                body: {
                    labels_to_remove: [{ id: labelToDelete.id }],
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            });
            setLabelToDelete(null);
        }
    };

    const handleCancelDeleteLabel = () => {
        deleteDialogState.close();
        setLabelToDelete(null);
        popoverState.open();
    };

    const handleSaveNewLabel = (name: string, color: string) => {
        updateLabelMutation.mutate({
            body: {
                labels_to_add: [{ name, color, hotkey: null }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const hasLabels = labels.filter((label) => label.id !== EMPTY_LABEL_ID).length > 0;

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            {hasLabels && (
                <div aria-label={'Labels'} className={classes.labelsContainer}>
                    {labels.map((label) => (
                        <Fragment key={label.id}>
                            {label.id === EMPTY_LABEL_ID && <Divider size={'S'} orientation={'vertical'} />}
                            <LabelBadge
                                label={label}
                                isSelected={isLabelSelected(label)}
                                onClick={() => handleLabelClick(label)}
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
                    <EditLabelsPopover
                        labels={labels}
                        onLabelSelect={handleLabelClick}
                        isLabelSelected={isLabelSelected}
                        onRequestDeleteLabel={handleRequestDeleteLabel}
                        onSaveNewLabel={handleSaveNewLabel}
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
