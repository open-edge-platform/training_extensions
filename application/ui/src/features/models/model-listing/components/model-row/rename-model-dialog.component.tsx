// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, TextField } from '@geti/ui';

interface RenameModelDialogProps {
    currentName: string;
    onRename: (newName: string) => void;
    onClose: () => void;
}

export const RenameModelDialog = ({ currentName, onRename, onClose }: RenameModelDialogProps) => {
    const [newName, setNewName] = useState(currentName);

    const isValid = newName.trim().length > 0;
    const hasSameName = newName.trim() === currentName;

    const handleRename = () => {
        onRename(newName.trim());
    };

    return (
        <Dialog>
            <Heading>Rename Model</Heading>
            <Divider />
            <Content>
                <TextField label='Model name' value={newName} onChange={setNewName} width='100%' />
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={handleRename} isDisabled={!isValid || hasSameName}>
                    Rename
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
