// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Flex, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';

import classes from './image-folder.module.scss';

export const ImageFolder = () => {
    const [route, setRoute] = useState('');

    return (
        <Flex direction='column' gap='size-200'>
            <Flex direction='row' gap='size-200'>
                <TextField
                    flex='1'
                    placeholder='Image folder path'
                    name='images_folder_path'
                    value={route}
                    onChange={setRoute}
                />

                <Flex alignItems={'center'} justifyContent={'center'} UNSAFE_className={classes.folderIcon}>
                    <Folder />
                </Flex>
            </Flex>

            <Button maxWidth={'size-1000'} isDisabled={isEmpty(route)}>
                Apply
            </Button>
        </Flex>
    );
};
