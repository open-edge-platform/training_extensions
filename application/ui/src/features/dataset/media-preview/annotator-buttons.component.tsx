// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, ToggleButton } from '@geti/ui';
import { useAnnotations } from 'src/features/annotator/annotations-provider.component';

type AnnotatorButtonsProps = {
    isFocussed: boolean;
    onFocus: (focussed: boolean) => void;
    onClose: () => void;
};
export const AnnotatorButtons = ({ isFocussed, onFocus, onClose }: AnnotatorButtonsProps) => {
    const { saveAnnotations, isSaving } = useAnnotations();

    return (
        <ButtonGroup>
            <Button
                variant='secondary'
                onPress={() => {
                    saveAnnotations();
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
