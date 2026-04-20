// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FocusEvent, KeyboardEvent, useRef, useState } from 'react';

import { ActionButton, DOMRefValue, Grid, TextField, useUnwrapDOMRef, View } from '@geti/ui';
import { Add, Close } from '@geti/ui/icons';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { getRandomDistinctColor } from '../../label-utils';

import classes from '../label-row/label-row.module.scss';

type NewLabelRowProps = {
    onSave: (name: string, color: string) => void;
    onCancel: () => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
};

export const NewLabelRow = ({ onSave, onCancel, validateName }: NewLabelRowProps) => {
    const rowRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const rowRefUnwrapped = useUnwrapDOMRef(rowRef);
    const [name, setName] = useState('');
    const [color, setColor] = useState(getRandomDistinctColor);

    const isEmptyName = name.trim().length === 0;
    const validationError = isEmptyName ? undefined : validateName(name);

    const isCreateButtonDisabled = isEmptyName || validationError !== undefined;

    const handleSave = () => {
        const trimmedName = name.trim();

        if (trimmedName.length > 0 && !validationError) {
            onSave(trimmedName, color);
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

        const trimmedName = name.trim();

        if (trimmedName.length > 0 && !validationError) {
            onSave(trimmedName, color);
        } else if (trimmedName.length === 0) {
            onCancel();
        }
    };

    return (
        <Grid
            ref={rowRef}
            columns={['size-350', 'size-400', '1fr', 'size-400', 'size-400']}
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color }}
        >
            <View />

            <LabelColorPicker color={color} onChange={setColor} />

            <View>
                <TextField
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
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
                isQuiet
                aria-label={'Create new label'}
                onPress={handleSave}
                isDisabled={isCreateButtonDisabled}
            >
                <Add />
            </ActionButton>

            <ActionButton
                aria-label='Cancel new label'
                isQuiet
                onPress={onCancel}
                UNSAFE_className={classes.deleteButton}
            >
                <Close />
            </ActionButton>
        </Grid>
    );
};
