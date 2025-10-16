// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useAnnotationActions } from 'src/features/annotator/annotation-actions-provider.component';

type AnnotatorButtonsProps = {
    onClose: () => void;
};

export const AnnotatorButtons = ({ onClose }: AnnotatorButtonsProps) => {
    const { submitAnnotations, isSaving } = useAnnotationActions();

    return (
        <ButtonGroup>
            <Button
                variant='secondary'
                onPress={() => {
                    submitAnnotations();
                    onClose();
                }}
                isDisabled={isSaving}
            >
                Save
            </Button>
            <Button variant='secondary' onPress={onClose}>
                Close
            </Button>
        </ButtonGroup>
    );
};
