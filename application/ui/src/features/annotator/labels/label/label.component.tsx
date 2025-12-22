// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ComponentProps,
    createContext,
    CSSProperties,
    FormEvent,
    KeyboardEvent,
    ReactNode,
    RefObject,
    use,
    useState,
} from 'react';

import { ActionButton, ColorPickerDialog, DOMRefValue, Form, TextField, TextFieldRef } from '@geti/ui';
import { isEmpty } from 'lodash-es';
import { Label as LabelType } from 'src/constants/shared-types';

import { MAX_LABEL_NAME_LENGTH, validateLabelName } from '../utils';

import styles from './label.module.scss';

interface LabelNameProps {
    isQuiet?: boolean;
    onClose: () => void;
    ariaLabel: string;
}

const autoFocus = (ref: TextFieldRef<HTMLInputElement> | null) => {
    if (ref === null) return;

    ref.focus();
};

const LabelName = ({ isQuiet, onClose, ariaLabel }: LabelNameProps) => {
    const { name, onNameChange, validationError } = useLabelContext();

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
            onClose();
        }
    };

    return (
        <TextField
            isQuiet={isQuiet}
            flex={1}
            ref={autoFocus}
            placeholder={'Label name'}
            aria-label={ariaLabel}
            value={name}
            onChange={onNameChange}
            maxLength={MAX_LABEL_NAME_LENGTH}
            onKeyDown={(e) => handleKeyDown(e)}
            errorMessage={validationError}
            validationState={validationError ? 'invalid' : undefined}
            isRequired
        />
    );
};

interface LabelButonProps {
    isDisabled?: boolean;
    color: string;
    children: ReactNode;
}

const LabelButton = ({ isDisabled, children, color }: LabelButonProps) => {
    const { isButtonDisabled } = useLabelContext();

    return (
        <ActionButton
            isQuiet
            type={'submit'}
            aria-label={'Confirm label'}
            isDisabled={isButtonDisabled || isDisabled}
            UNSAFE_style={
                {
                    '--labelButtonBgColor': color,
                } as CSSProperties
            }
            UNSAFE_className={styles.labelButton}
        >
            {children}
        </ActionButton>
    );
};

interface LabelColorPickerProps {
    onOpenChange?: (isOpen: boolean) => void;
}

const LabelColorPicker = ({ onOpenChange }: LabelColorPickerProps) => {
    const { color, onColorChange } = useLabelContext();

    return (
        <ColorPickerDialog
            color={color}
            id={'change-color-button'}
            data-testid={'change-color-button'}
            onColorChange={onColorChange}
            onOpenChange={onOpenChange}
            size={'M'}
        />
    );
};

interface LabelFormProps {
    onSubmit: (label: LabelType) => void;
    ref?: RefObject<DOMRefValue<HTMLFormElement> | null> | null;
    children: ComponentProps<typeof Form>['children'];
}

const LabelForm = ({ onSubmit, children, ref }: LabelFormProps) => {
    const { newLabel } = useLabelContext();

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        onSubmit(newLabel);
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit} ref={ref}>
            {children}
        </Form>
    );
};

interface LabelContextProps {
    isButtonDisabled: boolean;
    color: string;
    onColorChange: (color: string) => void;
    name: string;
    onNameChange: (name: string) => void;
    validationError?: string;
    newLabel: LabelType;
}

const LabelContext = createContext<LabelContextProps | null>(null);

interface LabelProps {
    children: ReactNode;
    label: LabelType;
    existingLabels: LabelType[];
}

export const Label = ({ children, label, existingLabels }: LabelProps) => {
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const validationError = validateLabelName(name, existingLabels, label.id);
    const hasSameName = name.trim() === label.name.trim();
    const hasSameColor = color === label.color;
    const isButtonDisabled = !!validationError || isEmpty(name.trim()) || (hasSameName && hasSameColor);

    const newLabel: LabelType = {
        id: label.id,
        name,
        color,
    };

    return (
        <LabelContext
            value={{
                color,
                onColorChange: setColor,
                name,
                onNameChange: setName,
                validationError,
                isButtonDisabled,
                newLabel,
            }}
        >
            {children}
        </LabelContext>
    );
};

Label.Form = LabelForm;
Label.NameField = LabelName;
Label.ColorPicker = LabelColorPicker;
Label.Button = LabelButton;

const useLabelContext = () => {
    const context = use(LabelContext);

    if (!context) {
        throw new Error('useLabelContext must be used within a LabelProvider');
    }

    return context;
};
