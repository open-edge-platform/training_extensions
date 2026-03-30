// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, TextField } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';
import { LocalFolderSinkConfig } from '../utils';

type LocalFolderProps = {
    defaultState: LocalFolderSinkConfig;
};

export const LocalFolder = ({ defaultState }: LocalFolderProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />

            <Flex gap='size-200'>
                <TextField label='Name' name='name' defaultValue={defaultState.name || 'Local folder sink'} />
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
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
