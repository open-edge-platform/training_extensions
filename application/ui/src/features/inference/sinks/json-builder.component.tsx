// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useState } from 'react';

import { ActionButton, Content, ContextualHelp, dimensionValue, Flex, Grid, Text, TextField } from '@geti/ui';
import { Add, Delete } from '@geti/ui/icons';

import classes from './json-builder.module.scss';

type JsonBuilderProps = {
    title: string;
    keysName: string;
    valuesName: string;
};

type Pair = Record<Fields, string>;

enum Fields {
    KEY = 'key',
    VALUE = 'value',
}

const updatePairAtIndex = (indexToUpdate: number, field: Fields, value: string) => (pair: Pair, index: number) =>
    index === indexToUpdate ? { ...pair, [field]: value } : pair;

export const JsonBuilder = ({ title, keysName, valuesName }: JsonBuilderProps) => {
    const [pairs, setPairs] = useState<Pair[]>([]);

    const addPair = () => {
        setPairs([...pairs, { key: '', value: '' }]);
    };

    const updatePair = (indexToUpdate: number, field: Fields, value: string) => {
        setPairs((prevValues) => prevValues.map(updatePairAtIndex(indexToUpdate, field, value)));
    };

    const removePair = (indexToRemove: number) => {
        setPairs((prevValues) => prevValues.filter((_, index) => index !== indexToRemove));
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex gap={'size-100'}>
                    <Text
                        UNSAFE_style={{
                            fontSize: dimensionValue('size-150'),
                            color: 'var(--spectrum-alias-label-text-color)',
                        }}
                    >
                        {title}
                    </Text>

                    <ContextualHelp variant='info'>
                        <Content>
                            <Text>
                                Add as many key-value pairs as needed. Each pair will be included in the `headers`
                                object.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
                <ActionButton isQuiet onPress={addPair} UNSAFE_className={classes.addButton}>
                    <Add />
                </ActionButton>
            </Flex>

            <Grid columns={['1fr', '1fr', '50px']} gap={'size-100'}>
                {pairs.map((pair, index) => (
                    <Fragment key={`pair-${index}`}>
                        <TextField
                            isQuiet
                            width={'100%'}
                            value={pair.key}
                            name={keysName}
                            placeholder='key'
                            onChange={(val) => updatePair(index, Fields.KEY, val)}
                        />
                        <TextField
                            isQuiet
                            width={'100%'}
                            value={pair.value}
                            name={valuesName}
                            placeholder='value'
                            onChange={(val) => updatePair(index, Fields.VALUE, val)}
                        />
                        <ActionButton
                            aria-label='Remove'
                            UNSAFE_className={classes.addButton}
                            onPress={() => removePair(index)}
                        >
                            <Delete />
                        </ActionButton>
                    </Fragment>
                ))}
            </Grid>
        </Flex>
    );
};
