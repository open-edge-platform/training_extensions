// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { Label } from 'src/constants/shared-types';
import { useAnnotationActions } from 'src/shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from 'src/shared/annotator/annotator-provider.component';

import { useUpdateLabel } from '../api/use-update-label';
import { EditLabel } from '../edit-label/edit-label.component';
import { LabelBadge } from '../label-badge/label-badge.component';

import classes from './label-list-item.module.scss';

interface LabelListItemViewProps {
    label: Label;
    onSelect: () => void;
    isSelected: boolean;
    onEdit: () => void;
}

const LabelListItemView = ({ label, onSelect, isSelected, onEdit }: LabelListItemViewProps) => {
    const projectId = useProjectIdentifier();

    const { setSelectedLabelId } = useAnnotator();

    const updateLabelMutation = useUpdateLabel();

    const deleteLabel = () => {
        updateLabelMutation.mutate(
            {
                body: {
                    labels_to_remove: [{ id: label.id }],
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: () => {
                    setSelectedLabelId(null);
                },
            }
        );
    };

    return (
        <LabelBadge onClick={onSelect} key={label.id} label={label} isSelected={isSelected}>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton
                    aria-label={`Edit ${label.name} label`}
                    isQuiet
                    UNSAFE_className={classes.iconButton}
                    onPress={onEdit}
                >
                    <Edit />
                </ActionButton>
                <Tooltip>Edit label name</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton
                    aria-label={`Delete ${label.name} label`}
                    isQuiet
                    UNSAFE_className={classes.iconButton}
                    onPress={deleteLabel}
                >
                    <Close />
                </ActionButton>
                <Tooltip>Delete label</Tooltip>
            </TooltipTrigger>
        </LabelBadge>
    );
};

interface LabelListItemProps {
    label: Label;
    onSelect: () => void;
    isSelected: boolean;
    existingLabels: Label[];
}

export const LabelListItem = ({ label, onSelect, isSelected, existingLabels }: LabelListItemProps) => {
    const [isInEdition, setIsInEdition] = useState<boolean>(false);
    const projectId = useProjectIdentifier();
    const { updateAnnotations, annotations } = useAnnotationActions();

    const updateLabelMutation = useUpdateLabel();
    const updateLabel = (newLabel: Label) => {
        updateLabelMutation.mutate(
            {
                body: {
                    labels_to_edit: [{ new_color: newLabel.color, new_name: newLabel.name, id: label.id }],
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: () => {
                    setIsInEdition(false);

                    if (label.color !== newLabel.color || label.name !== newLabel.name) {
                        const updatedAnnotations = annotations.map((annotation) => ({
                            ...annotation,
                            labels: annotation.labels.map((annotationLabel: Label) =>
                                annotationLabel.id === label.id
                                    ? { ...annotationLabel, color: newLabel.color, name: newLabel.name }
                                    : annotationLabel
                            ),
                        }));

                        updateAnnotations(updatedAnnotations);
                    }
                },
            }
        );
    };

    const handleClose = () => {
        setIsInEdition(false);
    };

    if (isInEdition) {
        return (
            <EditLabel
                shouldCloseOnOutsideClick
                onAccept={updateLabel}
                onClose={handleClose}
                label={label}
                width={'size-2400'}
                existingLabels={existingLabels}
                isDisabled={updateLabelMutation.isPending}
            />
        );
    }

    return (
        <LabelListItemView
            label={label}
            onSelect={onSelect}
            isSelected={isSelected}
            onEdit={() => setIsInEdition(true)}
        />
    );
};
