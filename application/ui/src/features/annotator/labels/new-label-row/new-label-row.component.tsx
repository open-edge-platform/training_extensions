// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FocusEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';

import { ActionButton, DOMRefValue, Flex, TextField, TextFieldRef, useUnwrapDOMRef, View } from '@geti/ui';
import { Close } from '@geti/ui/icons';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { validateLabelName } from '../../../../components/label-fields/label-validation';
import type { Label } from '../../../../constants/shared-types';
import { getRandomDistinctColor } from '../../label-utils';

import classes from '../edit-label-row/edit-label-row.module.scss';

interface NewLabelRowProps {
    existingLabels: Label[];
    onSave: (name: string, color: string) => void;
    onCancel: () => void;
}

export const NewLabelRow = ({ existingLabels, onSave, onCancel }: NewLabelRowProps) => {
    const rowRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const rowRefUnwrapped = useUnwrapDOMRef(rowRef);
    const inputRef = useRef<TextFieldRef<HTMLInputElement>>(null);
    const inputRefUnwrapped = useUnwrapDOMRef(inputRef);
    const [name, setName] = useState('');
    const [color, setColor] = useState(getRandomDistinctColor);

    const validationError = name.trim() === '' ? undefined : validateLabelName(name, existingLabels);

    useEffect(() => {
        // Focus the input when the component mounts
        inputRefUnwrapped.current?.focus();
    }, [inputRefUnwrapped]);

    const handleSave = () => {
        if (name.trim() !== '' && !validationError) {
            onSave(name.trim(), color);
        }
    };

    const handleNameKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            handleSave();
        } else if (event.key === 'Escape') {
            onCancel();
        }
    };

    const handleBlur = (event: FocusEvent<HTMLInputElement>) => {
        // Check if the blur target is within the row (e.g., clicking color picker)
        const relatedTarget = event.relatedTarget as Node | null;
        if (relatedTarget && rowRefUnwrapped.current?.contains(relatedTarget)) {
            return;
        }

        if (name.trim() !== '' && !validationError) {
            onSave(name.trim(), color);
        } else if (name.trim() === '') {
            onCancel();
        }
    };

    return (
        <Flex
            ref={rowRef}
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color }}
        >
            <View
                UNSAFE_style={{
                    width: 'calc(var(--spectrum-global-dimension-size-300) + var(--spectrum-global-dimension-size-75))',
                }}
            />

            <LabelColorPicker color={color} onChange={setColor} />

            <View flex={1}>
                <TextField
                    ref={inputRef}
                    aria-label={'New label name'}
                    placeholder={'Label name'}
                    value={name}
                    onChange={setName}
                    onKeyDown={handleNameKeyDown}
                    onBlur={handleBlur}
                    width={'100%'}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                />
            </View>

            <ActionButton
                aria-label='Cancel new label'
                isQuiet
                onPress={onCancel}
                UNSAFE_className={classes.deleteButton}
            >
                <Close />
            </ActionButton>
        </Flex>
    );
};
