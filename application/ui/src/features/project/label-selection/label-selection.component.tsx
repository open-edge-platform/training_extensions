// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import {
    ActionButton,
    Button,
    ColorEditor,
    ColorSwatch,
    ColorSwatchPicker,
    Flex,
    Grid,
    ColorPicker as SpectrumColorPicker,
    SpectrumColorPickerProps,
    Text,
    toast,
} from '@geti/ui';
import { Add, Delete } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import { Label } from '../../annotator/types';
import { LabelItemProps } from './interface';

import classes from './label-selection.module.scss';

const ColorPicker = ({ onChange, value }: SpectrumColorPickerProps) => {
    return (
        <SpectrumColorPicker value={value} onChange={onChange} rounding={'none'}>
            <Flex direction='column' gap='size-300'>
                <ColorEditor />
                <ColorSwatchPicker>
                    <ColorSwatch color='#E91E63' />
                    <ColorSwatch color='#9C27B0' />
                    <ColorSwatch color='#2196F3' />
                    <ColorSwatch color='#4CAF50' />
                    <ColorSwatch color='#FFEB3B' />
                    <ColorSwatch color='#FF9800' />
                    <ColorSwatch color='#000000' />
                </ColorSwatchPicker>
            </Flex>
        </SpectrumColorPicker>
    );
};

const LabelInput = ({ value, onChange }: { value: string; onChange: (newValue: string) => void }) => {
    return (
        <input
            name={`Label input for ${value}`}
            aria-label={`Label input for ${value}`}
            className={classes.labelInput}
            type='text'
            value={value}
            onChange={(e) => onChange(e.target.value)}
        />
    );
};

const LabelItem = ({ label, onDelete, onUpdate }: LabelItemProps) => {
    const { id, name, color } = label;

    return (
        <Grid columns={['size-400', '1fr', 'size-400']} gap={'size-50'} maxWidth={'640px'} width={'100%'} id={id}>
            <ColorPicker
                onChange={(newColor) => {
                    onUpdate({ ...label, color: newColor.toString() });
                }}
                value={color ?? undefined}
            />
            <LabelInput
                value={name}
                onChange={(newName) => {
                    onUpdate({ ...label, name: newName });
                }}
            />
            <Flex justifyContent={'center'} alignItems={'center'}>
                <ActionButton
                    aria-label={`Delete label ${name}`}
                    onPress={() => onDelete(id)}
                    UNSAFE_className={classes.deleteButton}
                >
                    <Delete fill='white' />
                </ActionButton>
            </Flex>
        </Grid>
    );
};

type LabelSelectionProps = {
    labels: Label[];
    setLabels: Dispatch<SetStateAction<Label[]>>;
};
export const LabelSelection = ({ labels, setLabels }: LabelSelectionProps) => {
    const handleDeleteItem = (id: string) => {
        if (labels.length > 1) {
            setLabels(labels.filter((label) => label.id !== id));
        } else {
            toast({ type: 'info', message: 'At least one object is required' });
        }
    };

    const handleAddItem = () => {
        setLabels([
            ...labels,
            {
                id: uuid(),
                color: '',
                name: 'Object',
            },
        ]);
    };

    const handleUpdateItem = (updatedLabel: Label) => {
        const updatedLabels = labels.map((label) => (label.id === updatedLabel.id ? updatedLabel : label));

        setLabels(updatedLabels);
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
            <Flex direction={'column'} alignItems={'center'} gap={'size-100'} width={'100%'}>
                {labels.map((label) => {
                    return (
                        <LabelItem
                            key={label.id}
                            label={label}
                            onDelete={handleDeleteItem}
                            onUpdate={handleUpdateItem}
                        />
                    );
                })}
            </Flex>

            <Flex gap={'size-200'}>
                <Button width={'size-2000'} variant={'secondary'} onPress={handleAddItem}>
                    <Text>Add next object</Text>
                    <Add fill='white' />
                </Button>
            </Flex>
        </Flex>
    );
};
