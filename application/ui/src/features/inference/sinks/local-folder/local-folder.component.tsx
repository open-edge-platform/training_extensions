// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, TextField } from '@geti/ui';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';
import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';
import { LocalFolderSinkConfig } from '../utils';

import classes from './local-folder.module.scss';

type LocalFolderProps = {
    defaultState: LocalFolderSinkConfig;
};

export const LocalFolder = ({ defaultState }: LocalFolderProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />

            <Flex gap='size-200'>
                <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />
            </Flex>

            <Flex>
                <RateLimitFields rateLimit={defaultState.rate_limit} />
            </Flex>

            <Flex gap='size-50'>
                <TextField
                    isRequired
                    width={'100%'}
                    label='Folder Path'
                    name='folder_path'
                    defaultValue={defaultState.folder_path}
                />

                <Flex
                    alignSelf={'end'}
                    height={'size-400'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    UNSAFE_className={classes.folderIcon}
                >
                    <Folder />
                </Flex>
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
