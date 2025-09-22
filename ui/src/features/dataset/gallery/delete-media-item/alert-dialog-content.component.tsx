// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Flex, Text } from '@geti/ui';

import { useEventListener } from '../../../../hooks/event-listener/event-listener.hook';

type AlertDialogContentProps = {
    itemsIds: string[];
    onPrimaryAction: () => void;
};

export const AlertDialogContent = ({ itemsIds, onPrimaryAction }: AlertDialogContentProps) => {
    useEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            onPrimaryAction();
        }
    });

    return (
        <AlertDialog
            maxHeight={'size-6000'}
            title='Delete Items'
            variant='confirmation'
            primaryActionLabel='Confirm'
            secondaryActionLabel='Close'
            onPrimaryAction={onPrimaryAction}
        >
            <Text>Are you sure you want to delete the next items?</Text>

            <Flex direction={'column'} marginTop={'size-100'}>
                {itemsIds.map((itemId) => (
                    <Text key={itemId}>- {itemId}</Text>
                ))}
            </Flex>
        </AlertDialog>
    );
};
