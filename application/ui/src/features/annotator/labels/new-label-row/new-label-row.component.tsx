// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FocusEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';

import { ActionButton, DOMRefValue, Flex, TextField, TextFieldRef, useUnwrapDOMRef, View } from '@geti/ui';
import { Close } from '@geti/ui/icons';

import { HotkeyField } from '../../../../components/label-fields/hotkey-field.component';
import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { validateLabelName } from '../../../../components/label-fields/label-validation';
import type { Label } from '../../../../constants/shared-types';
import { getRandomDistinctColor } from '../../label-utils';

import classes from '../edit-label-row/edit-label-row.module.scss';

interface NewLabelRowProps {
    existingLabels: Label[];
    allHotkeys: string[];
    onSave: (name: string, color: string, hotkey: string | null) => void;
    onCancel: () => void;
}

export const NewLabelRow = ({ existingLabels, allHotkeys, onSave, onCancel }: NewLabelRowProps) => {
    const rowRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const rowRefUnwrapped = useUnwrapDOMRef(rowRef);
    const inputRef = useRef<TextFieldRef<HTMLInputElement>>(null);
    const inputRefUnwrapped = useUnwrapDOMRef(inputRef);
    const [name, setName] = useState('');
    const [color, setColor] = useState(getRandomDistinctColor);
    const [hotkey, setHotkey] = useState<string | null>(null);

    const validationError = name.trim() === '' ? undefined : validateLabelName(name, existingLabels);

    useEffect(() => {
        // Focus the input when the component mounts
        inputRefUnwrapped.current?.focus();
    }, [inputRefUnwrapped]);

    const handleSave = () => {
        if (name.trim() !== '' && !validationError) {
            onSave(name.trim(), color, hotkey);
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
        // Check if the blur target is within the row (e.g., clicking color picker or hotkey field)
        const relatedTarget = event.relatedTarget as Node | null;
        if (relatedTarget && rowRefUnwrapped.current?.contains(relatedTarget)) {
            return;
        }

        if (name.trim() !== '' && !validationError) {
            onSave(name.trim(), color, hotkey);
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

            <View width={'size-1600'}>
                <HotkeyField hotkey={hotkey} onHotkeyChange={setHotkey} allHotkeys={allHotkeys} />
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
