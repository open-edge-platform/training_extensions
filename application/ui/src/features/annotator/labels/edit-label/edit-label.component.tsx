// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { DimensionValue, DOMRefValue, Flex, useUnwrapDOMRef } from '@geti/ui';
import { Checkmark } from '@geti/ui/icons';
import { Label as LabelType } from 'src/constants/shared-types';
import { useOnOutsideClick } from 'src/shared/hooks/use-on-click-outside.hook';

import { Label } from '../label/label.component';

import classes from './edit-label.module.scss';

interface EditLabelProps {
    label: LabelType;
    onAccept: (editedLabel: LabelType) => void;
    onClose: () => void;
    width?: DimensionValue;
    isDisabled?: boolean;
    existingLabels: LabelType[];
    shouldCloseOnOutsideClick?: boolean;
}

const useCloseLabelEditionOnOutsideClick = ({
    onClose,
    shouldCloseOnOutsideClick,
    isColorPickerOpen,
}: {
    onClose: () => void;
    shouldCloseOnOutsideClick: boolean;
    isColorPickerOpen: boolean;
}) => {
    const wrappedFormRef = useRef<DOMRefValue<HTMLFormElement>>(null);
    const formRef = useUnwrapDOMRef(wrappedFormRef);

    useOnOutsideClick(formRef, () => {
        if (isColorPickerOpen) {
            return;
        }

        onClose();
    });

    return shouldCloseOnOutsideClick ? wrappedFormRef : null;
};

export const EditLabel = ({
    label,
    onAccept,
    onClose,
    width,
    isDisabled,
    existingLabels,
    shouldCloseOnOutsideClick = false,
}: EditLabelProps) => {
    const [isColorPickerOpen, setIsColorPickerOpen] = useState<boolean>(false);

    const formRef = useCloseLabelEditionOnOutsideClick({ onClose, shouldCloseOnOutsideClick, isColorPickerOpen });

    return (
        <Label label={label} existingLabels={existingLabels}>
            <Label.Form onSubmit={onAccept} ref={formRef}>
                <Flex
                    marginTop={0}
                    gap={'size-50'}
                    width={width}
                    justifyContent={'center'}
                    UNSAFE_className={classes.editLabelContainer}
                >
                    <Label.ColorPicker onOpenChange={setIsColorPickerOpen} />

                    <Label.NameField isQuiet onClose={onClose} ariaLabel={'Edit label name'} />

                    <Label.Button isDisabled={isDisabled} color={'var(--energy-blue)'}>
                        <Checkmark />
                    </Label.Button>
                </Flex>
            </Label.Form>
        </Label>
    );
};
