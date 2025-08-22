// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Heading, Text } from '@geti/ui';

import { useSelectedData } from '../../../routes/dataset/provider';
import { CheckboxInput } from '../checkbox-input';
import { response } from '../mock-response';
import { toggleMultipleSelection, updateSelectedKeysTo } from './util';

export const Toolbar = () => {
    const { selectedKeys, setSelectedKeys, setMediaState } = useSelectedData();
    const totalSelectedElements = selectedKeys instanceof Set ? selectedKeys.size : 0;
    const hasSelectedElements = totalSelectedElements > 0;

    const message = hasSelectedElements ? `${totalSelectedElements} selected` : `${response.items.length} images`;

    const handleToggleManyItemSelection = () => {
        const images = response.items.map((item) => item.id);
        setSelectedKeys(toggleMultipleSelection(images));
    };

    const handleAccept = () => {
        setSelectedKeys(new Set());
        setMediaState(updateSelectedKeysTo(selectedKeys, 'accepted'));
    };

    const handleReject = () => {
        setSelectedKeys(new Set());
        setMediaState(updateSelectedKeysTo(selectedKeys, 'rejected'));
    };

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Heading level={1}>Data collection</Heading>
            <Divider size='S' />

            <Flex direction={'row'} alignItems={'center'} justifyContent={'space-between'}>
                <Flex
                    gap={'size-200'}
                    height={'size-400'}
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                >
                    <CheckboxInput
                        name={'select all'}
                        onChange={handleToggleManyItemSelection}
                        isChecked={totalSelectedElements === response.items.length}
                    />

                    <Divider orientation={'vertical'} size={'S'} />

                    {hasSelectedElements && (
                        <>
                            <Button variant={'accent'} onPress={handleAccept}>
                                Accept
                            </Button>
                            <Button variant={'secondary'} onPress={handleReject}>
                                Decline
                            </Button>
                        </>
                    )}
                </Flex>

                <Text>{message}</Text>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
