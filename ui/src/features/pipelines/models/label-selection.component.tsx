// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    ColorSwatch,
    ColorSwatchPicker,
    ColorPicker as SpectrumColorPicker,
    SpectrumColorPickerProps,
} from '@adobe/react-spectrum';
import { ActionButton, Button, ColorEditor, Flex, Grid, Text, toast } from '@geti/ui';
import { Add, Delete } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import classes from './label-selection.module.scss';

const ColorPicker = ({ onChange, value }: SpectrumColorPickerProps) => {
    return (
        <SpectrumColorPicker value={value} onChange={onChange}>
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
            aria-label={`Label input for ${value}`}
            className={classes.labelInput}
            type='text'
            value={value}
            onChange={(e) => onChange(e.target.value)}
        />
    );
};

type LabelItemProps = { id: string; colorValue: string; nameValue: string; onDelete: (id: string) => void };
const LabelItem = ({ id, colorValue, nameValue, onDelete }: LabelItemProps) => {
    const [color, setColor] = useState<string>(colorValue);
    const [name, setName] = useState<string>(nameValue);

    return (
        <Grid columns={['size-400', '1fr', 'size-400']} gap={'size-50'} maxWidth={'640px'} width={'100%'} id={id}>
            <ColorPicker
                onChange={(newColor) => {
                    setColor(newColor.toString());
                }}
                value={color ?? undefined}
            />
            <LabelInput
                value={name}
                onChange={(newValue) => {
                    setName(newValue);
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

export const LabelSelection = () => {
    const [items, setItems] = useState<Omit<LabelItemProps, 'onDelete'>[]>([
        { id: uuid(), colorValue: '#F20004', nameValue: 'Car' },
    ]);

    const handleDeleteItem = (id: string) => {
        if (items.length > 1) {
            setItems(items.filter((item) => item.id !== id));
        } else {
            toast({ type: 'info', message: 'At least one object is required' });
        }
    };

    const handleAddItem = () => {
        setItems([
            ...items,
            {
                id: uuid(),
                colorValue: '',
                nameValue: 'Object',
            },
        ]);
    };

    return (
        <Flex direction={'column'} alignItems={'center'} width={'100%'}>
            <Flex direction={'column'} alignItems={'center'} gap={'size-100'} width={'100%'}>
                {items.map((item) => {
                    return <LabelItem key={item.id} onDelete={handleDeleteItem} {...item} />;
                })}
            </Flex>

            <Flex gap={'size-200'}>
                <Button width={'size-2000'} variant={'secondary'} marginTop={'size-400'} onPress={handleAddItem}>
                    <Text>Add next object</Text>
                    <Add fill='white' />
                </Button>
            </Flex>
        </Flex>
    );
};
