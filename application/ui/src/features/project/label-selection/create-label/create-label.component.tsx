// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react';

import {
    ActionButton,
    ColorEditor,
    ColorSwatch,
    ColorSwatchPicker,
    DOMRefValue,
    Flex,
    Grid,
    ColorPicker as SpectrumColorPicker,
    SpectrumColorPickerProps,
    TextField,
    View,
} from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import type { Label, TaskType } from '../../../../constants/shared-types';
import { TASK_HOTKEYS } from '../../../../shared/hotkeys-definition';
import { validateLabelName } from '../validator';
import { HotkeyField } from './hotkey-field.component';

const PRESET_COLORS = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FFEB3B', '#FF9800', '#000000'];

const getRandomColor = () => {
    return PRESET_COLORS[Math.floor(Math.random() * PRESET_COLORS.length)];
};

const ColorPicker = ({ onChange, value }: SpectrumColorPickerProps) => {
    return (
        <SpectrumColorPicker value={value} onChange={onChange} rounding={'none'}>
            <Flex direction='column' gap='size-300'>
                <ColorEditor />
                <ColorSwatchPicker>
                    {PRESET_COLORS.map((color) => {
                        return <ColorSwatch color={color} key={color} />;
                    })}
                </ColorSwatchPicker>
            </Flex>
        </SpectrumColorPicker>
    );
};

const getInitialLabel = (): Label => ({ id: uuid(), color: getRandomColor(), name: '' });

export type CreateLabelProps = {
    onCreate: (label: Label) => void;
    labels: Label[];
    taskType: TaskType;
};

export const CreateLabel = ({ labels, onCreate, taskType }: CreateLabelProps) => {
    const [newLabel, setNewLabel] = useState<Label>(getInitialLabel);
    const ref = useRef<DOMRefValue<HTMLDivElement>>(null);

    const labelsHotkeys = labels.map((label) => label.hotkey).filter((hotkey) => hotkey != null);
    const appHotkeys = Object.values(TASK_HOTKEYS[taskType]);
    const allHotkeys = [...labelsHotkeys, ...appHotkeys];

    const validationResult = validateLabelName(newLabel, labels);
    const isCreateLabelDisabled = newLabel.name.trim().length === 0 || validationResult !== undefined;

    const createLabel = useCallback(() => {
        if (isCreateLabelDisabled) return;

        onCreate(newLabel);
        setNewLabel(getInitialLabel);
    }, [onCreate, newLabel, isCreateLabelDisabled]);

    useEffect(() => {
        const node = ref.current?.UNSAFE_getDOMNode();
        if (node == null) return;

        const abortController = new AbortController();

        node.addEventListener(
            'keydown',
            (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();

                    createLabel();
                }
            },
            {
                signal: abortController.signal,
            }
        );

        return () => {
            abortController.abort();
        };
    }, [createLabel]);

    return (
        <Grid
            columns={['size-400', '1fr', 'size-1600', 'size-400']}
            gap={'size-50'}
            maxWidth={'640px'}
            width={'100%'}
            alignItems={'start'}
            ref={ref}
        >
            <ColorPicker
                onChange={(newColor) => {
                    setNewLabel((prevLabel) => ({ ...prevLabel, color: newColor.toString() }));
                }}
                value={newLabel?.color}
            />
            <View>
                <TextField
                    aria-label={'Create label input'}
                    placeholder={'Create label'}
                    value={newLabel?.name}
                    onChange={(newName) => setNewLabel((prevLabel) => ({ ...prevLabel, name: newName }))}
                    errorMessage={validationResult}
                    validationState={validationResult === undefined ? undefined : 'invalid'}
                    width={'100%'}
                />
            </View>
            <View>
                <HotkeyField
                    hotkey={newLabel.hotkey}
                    onHotkeyChange={(newHotkey) => setNewLabel((prevLabel) => ({ ...prevLabel, hotkey: newHotkey }))}
                    allHotkeys={allHotkeys}
                />
            </View>

            <ActionButton
                isQuiet
                onPress={createLabel}
                isDisabled={isCreateLabelDisabled}
                aria-label={`Create label ${newLabel?.name}`}
            >
                <Add />
            </ActionButton>
        </Grid>
    );
};
