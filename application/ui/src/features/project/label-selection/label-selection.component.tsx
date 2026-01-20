// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useRef, useState } from 'react';

import {
    ActionButton,
    ColorEditor,
    ColorSwatch,
    ColorSwatchPicker,
    Flex,
    Grid,
    ColorPicker as SpectrumColorPicker,
    SpectrumColorPickerProps,
    TextField,
    toast,
    View,
} from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import type { Label, TaskType } from '../../../constants/shared-types';
import { TASK_HOTKEYS } from '../../../shared/hotkeys-definition';
import { HotkeyField } from './hotkey-field.component';
import { LabelTag } from './label-tag/label-tag.component';
import { validateLabelName } from './validator';

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

const CreateLabel = ({ labels, onCreate, taskType }: CreateLabelProps) => {
    const [newLabel, setNewLabel] = useState<Label>(getInitialLabel);
    const isDirty = useRef<boolean>(false);

    const labelsHotkeys = labels.map((label) => label.hotkey).filter((hotkey) => hotkey != null);
    const appHotkeys = Object.values(TASK_HOTKEYS[taskType]);
    const allHotkeys = [...labelsHotkeys, ...appHotkeys];

    const validationResult = isDirty.current ? validateLabelName(newLabel, labels) : undefined;
    const isCreateLabelDisabled = validationResult !== undefined;

    const createLabel = () => {
        isDirty.current = false;

        onCreate(newLabel);
        setNewLabel(getInitialLabel);
    };

    return (
        <Grid
            columns={['size-400', '1fr', 'size-1600', 'size-400']}
            gap={'size-50'}
            maxWidth={'640px'}
            width={'100%'}
            alignItems={'start'}
        >
            <ColorPicker
                onChange={(newColor) => {
                    setNewLabel((prevLabel) => ({ ...prevLabel, color: newColor.toString() }));
                }}
                value={newLabel?.color}
            />
            <View>
                <TextField
                    onFocus={() => {
                        isDirty.current = true;
                    }}
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

type LabelSelectionProps = {
    labels: Label[];
    setLabels: Dispatch<SetStateAction<Label[]>>;
    taskType: TaskType;
};

export const LabelSelection = ({ labels, setLabels, taskType }: LabelSelectionProps) => {
    const handleDeleteItem = (id: string) => {
        const newLabels = labels.filter((label) => label.id !== id);
        setLabels(newLabels);

        if (newLabels.length === 0) {
            toast({ type: 'info', message: 'At least one object is required' });
        }
    };

    const handleAddItem = (label: Label) => {
        setLabels([...labels, label]);
    };

    return (
        <Flex
            direction={'column'}
            alignItems={'center'}
            height={'100%'}
            width={'100%'}
            gap={'size-300'}
            UNSAFE_style={{ overflow: 'auto' }}
        >
            <CreateLabel onCreate={handleAddItem} labels={labels} taskType={taskType} />
            <Flex gap={'size-100'} width={'100%'} wrap={'wrap'}>
                {labels.map((label) => (
                    <LabelTag key={label.id} label={label} onDelete={handleDeleteItem} />
                ))}
            </Flex>
        </Flex>
    );
};
