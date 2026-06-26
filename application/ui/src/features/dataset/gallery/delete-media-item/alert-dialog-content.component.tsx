// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Text } from '@geti-ui/ui';

import { pluralizeItems } from '../../../../shared/util';

type AlertDialogContentProps = {
    itemsIds: string[];
    onPrimaryAction: () => void;
};

export const AlertDialogContent = ({ itemsIds, onPrimaryAction }: AlertDialogContentProps) => {
    return (
        <AlertDialog
            maxHeight={'size-6000'}
            title='Delete Items'
            variant='destructive'
            primaryActionLabel='Confirm'
            secondaryActionLabel='Cancel'
            onPrimaryAction={onPrimaryAction}
            autoFocusButton='primary'
        >
            <Text>{`Are you sure you want to delete ${itemsIds.length} ${pluralizeItems(itemsIds.length)}?`}</Text>
        </AlertDialog>
    );
};
