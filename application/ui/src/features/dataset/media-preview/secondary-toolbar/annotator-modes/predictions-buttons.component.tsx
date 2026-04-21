// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Icon, Text } from '@geti/ui';
import { Checkmark, Edit } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';
import { convertPredictionToAnnotation } from '../../../../annotator/annotations/utils';

type EditPredictionButtonProps = {
    isDisabled: boolean;
    onEditPrediction: () => void;
};

const EditPredictionButton = ({ isDisabled, onEditPrediction }: EditPredictionButtonProps) => {
    return (
        <ActionButton isQuiet onPress={onEditPrediction} isDisabled={isDisabled} aria-label={'Edit prediction'}>
            <Icon>
                <Edit />
            </Icon>
            <Text>Edit</Text>
        </ActionButton>
    );
};

type PredictionButtonsProps = {
    onSubmit: () => void;
    isDisabled: boolean;
    onModeChange: (mode: AnnotatorMode) => void;
};

export const PredictionButtons = ({ onSubmit, onModeChange, isDisabled }: PredictionButtonsProps) => {
    const { replaceAnnotations, annotations } = useAnnotationActions();

    const handleEditPrediction = () => {
        onModeChange('annotation');
        replaceAnnotations(annotations.map(convertPredictionToAnnotation));
    };

    return (
        <>
            <ActionButton isQuiet onPress={onSubmit} isDisabled={isDisabled}>
                <Checkmark />
                <Text>Confirm prediction</Text>
            </ActionButton>

            <EditPredictionButton onEditPrediction={handleEditPrediction} isDisabled={isDisabled} />
        </>
    );
};
