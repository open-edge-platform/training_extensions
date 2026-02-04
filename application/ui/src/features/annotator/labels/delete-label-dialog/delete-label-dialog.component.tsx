// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactElement } from 'react';

import { AlertDialog, DialogTrigger } from '@geti/ui';

import type { Label } from '../../../../constants/shared-types';

interface DeleteLabelDialogProps {
    label: Label;
    onDelete: () => void;
    children: ReactElement;
}

export const DeleteLabelDialog = ({ label, onDelete, children }: DeleteLabelDialogProps) => {
    return (
        <DialogTrigger>
            {children}
            {(close) => (
                <AlertDialog
                    title={'Delete label'}
                    variant={'destructive'}
                    primaryActionLabel={'Delete'}
                    cancelLabel={'Cancel'}
                    onPrimaryAction={() => {
                        onDelete();
                        close();
                    }}
                    onCancel={close}
                >
                    {`If you remove the '${label.name}' label, all annotations in your dataset that have `}
                    {"this label will be deleted. However, this won't impact any models you've trained in the past."}
                </AlertDialog>
            )}
        </DialogTrigger>
    );
};
