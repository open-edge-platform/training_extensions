// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useState } from 'react';

import { ActionButton, Content, ContextualHelp, dimensionValue, Flex, Grid, Text } from '@geti/ui';
import { Add, Delete } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { RequiredTextField } from '../../../components/required-text-field/required-text-field.component';

type KeyValueBuilderProps = {
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

export const KeyValueBuilder = ({ title, keysName, valuesName }: KeyValueBuilderProps) => {
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
                                Add as many key-value pairs as needed. Each pair will be included in the &apos;{title}
                                &apos; object.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
                <ActionButton isQuiet onPress={addPair}>
                    <Add />
                </ActionButton>
            </Flex>

            <Grid columns={['1fr', '1fr', '50px']} gap={'size-100'}>
                {pairs.map((pair, index) => (
                    <Fragment key={`pair-${index}`}>
                        <RequiredTextField
                            isQuiet
                            width={'100%'}
                            value={pair.key}
                            name={keysName}
                            placeholder='key'
                            errorMessage='Key cannot be empty'
                            onChange={(val) => updatePair(index, Fields.KEY, val)}
                        />
                        <RequiredTextField
                            isQuiet
                            width={'100%'}
                            value={pair.value}
                            name={valuesName}
                            placeholder='value'
                            errorMessage='Value cannot be empty'
                            isDisabled={isEmpty(pair.key)}
                            onChange={(val) => updatePair(index, Fields.VALUE, val)}
                        />
                        <ActionButton aria-label='Remove' onPress={() => removePair(index)}>
                            <Delete />
                        </ActionButton>
                    </Fragment>
                ))}
            </Grid>
        </Flex>
    );
};
