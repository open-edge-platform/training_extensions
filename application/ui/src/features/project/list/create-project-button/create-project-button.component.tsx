// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text } from '@geti/ui';

import { CreateProjectMenu } from './create-project-menu.component';

import classes from './create-project-button.module.scss';

interface CreateProjectButtonProps {
    buttonText: string;
    handleOpenDialog: () => void;
    onCreateFromDataset: () => void;
}

export const CreateProjectButton = ({
    buttonText,
    handleOpenDialog,
    onCreateFromDataset,
}: CreateProjectButtonProps) => {
    return (
        <Flex alignItems={'center'} gap={'size-10'}>
            <Button
                variant='accent'
                id='create-new-project-button'
                onPress={handleOpenDialog}
                UNSAFE_className={classes.createProjectButton}
            >
                <Text marginX='size-100' UNSAFE_style={{ whiteSpace: 'nowrap' }}>
                    {buttonText}
                </Text>
            </Button>
            <CreateProjectMenu onCreateFromDataset={onCreateFromDataset} />
        </Flex>
    );
};
