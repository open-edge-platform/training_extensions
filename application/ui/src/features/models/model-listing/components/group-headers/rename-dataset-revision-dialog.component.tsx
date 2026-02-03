// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Form, Heading, TextField } from '@geti/ui';

interface RenameDatasetRevisionDialogProps {
    currentName: string;
    onRename: (newName: string) => void;
    onClose: () => void;
    isPending?: boolean;
}

export const RenameDatasetRevisionDialog = ({
    currentName,
    onRename,
    onClose,
    isPending,
}: RenameDatasetRevisionDialogProps) => {
    const [newName, setNewName] = useState(currentName);

    const hasSameName = newName.trim() === currentName;

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        onRename(newName.trim());
    };

    return (
        <Dialog>
            <Heading>Rename dataset revision</Heading>

            <Divider />

            <Content>
                <Form onSubmit={handleSubmit} validationBehavior={'native'}>
                    <TextField
                        label={'Dataset revision name'}
                        value={newName}
                        onChange={setNewName}
                        width={'100%'}
                        isRequired
                    />
                    <ButtonGroup align={'end'} marginTop={'size-300'}>
                        <Button variant={'secondary'} onPress={onClose}>
                            Cancel
                        </Button>
                        <Button variant={'accent'} type={'submit'} isPending={isPending} isDisabled={hasSameName}>
                            Rename
                        </Button>
                    </ButtonGroup>
                </Form>
            </Content>
        </Dialog>
    );
};
