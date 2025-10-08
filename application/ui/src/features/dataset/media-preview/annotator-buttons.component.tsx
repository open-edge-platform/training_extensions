// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, ToggleButton } from '@geti/ui';
import { useAnnotationActions } from 'src/features/annotator/annotation-actions-provider.component';

type AnnotatorButtonsProps = {
    isFocussed: boolean;
    onFocus: (focussed: boolean) => void;
    onClose: () => void;
};
export const AnnotatorButtons = ({ isFocussed, onFocus, onClose }: AnnotatorButtonsProps) => {
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
            <ToggleButton marginEnd={'size-100'} isEmphasized isSelected={isFocussed} onChange={onFocus}>
                Focus
            </ToggleButton>
            <Button variant='secondary' onPress={onClose}>
                Close
            </Button>
        </ButtonGroup>
    );
};
