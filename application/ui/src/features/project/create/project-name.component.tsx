// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Text,
    TextField,
} from '@geti/ui';
import { Edit } from '@geti/ui/icons';

type ProjectNameProps = { name: string; setName: Dispatch<SetStateAction<string>> };
export const ProjectName = ({ name, setName }: ProjectNameProps) => {
    return (
        <DialogTrigger>
            <ActionButton isQuiet>
                <Heading level={4}>
                    <Flex alignItems={'center'} gap='size-200'>
                        <Text>{name}</Text>
                        <Edit fill={'white'} />
                    </Flex>
                </Heading>
            </ActionButton>
            {(close) => (
                <Dialog>
                    <Heading>Edit Project Name</Heading>
                    <Divider />

                    <Content>
                        <TextField aria-label={'edit project name'} value={name} onChange={setName} />
                    </Content>

                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            Cancel
                        </Button>
                        <Button variant='accent' onPress={close}>
                            Confirm
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
