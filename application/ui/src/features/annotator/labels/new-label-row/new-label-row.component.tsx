// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FocusEvent, KeyboardEvent, useRef, useState } from 'react';

import { ActionButton, Grid, TextField, View } from '@geti-ui/ui';
import { Close } from '@geti-ui/ui/icons';

import { LabelColorPicker } from '../../../../components/label-fields/label-color-picker.component';
import { getRandomDistinctColor } from '../../label-utils';

import classes from '../label-row/label-row.module.scss';

type NewLabelRowProps = {
    onSave: (name: string, color: string) => void;
    onCancel: () => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
};

export const NewLabelRow = ({ onSave, onCancel, validateName }: NewLabelRowProps) => {
    const rowRef = useRef<HTMLDivElement>(null);
    const [name, setName] = useState('');
    const [color, setColor] = useState(getRandomDistinctColor);

    const validationError = name.trim() === '' ? undefined : validateName(name);

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
        if (relatedTarget && rowRef.current?.contains(relatedTarget)) {
            return;
        }

        if (name.trim() !== '' && !validationError) {
            onSave(name.trim(), color);
        } else if (name.trim() === '') {
            onCancel();
        }
    };

    return (
        <div ref={rowRef} className={classes.labelRow} style={{ '--label-color': color } as React.CSSProperties}>
            <Grid
                columns={['size-350', 'size-400', '1fr', 'size-400', 'size-400']}
                gap={'size-100'}
                alignItems={'start'}
            >
                <View />

                <LabelColorPicker color={color} onChange={setColor} />

                <View>
                    <TextField
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

                <View />

                <ActionButton
                    aria-label='Cancel new label'
                    isQuiet
                    onPress={onCancel}
                    UNSAFE_className={classes.deleteButton}
                >
                    <Close />
                </ActionButton>
            </Grid>
        </div>
    );
};
