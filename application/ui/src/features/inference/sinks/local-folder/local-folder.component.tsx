// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, NumberField, TextField } from '@geti/ui';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';
import { OutputFormats } from '../output-formats/output-formats.component';
import { LocalFolderSinkConfig } from '../utils';

import classes from './local-folder.module.scss';

type LocalFolderProps = {
    defaultState: LocalFolderSinkConfig;
};

export const LocalFolder = ({ defaultState }: LocalFolderProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />

            <Flex direction={'row'} gap='size-200'>
                <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />
                <NumberField
                    label='Rate Limit'
                    name='rate_limit'
                    minValue={0}
                    step={0.1}
                    defaultValue={defaultState.rate_limit ?? undefined}
                />
            </Flex>

            <Flex direction='row' gap='size-200'>
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
