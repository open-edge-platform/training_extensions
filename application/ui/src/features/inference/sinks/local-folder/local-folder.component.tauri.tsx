// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react';

import { Flex, TextField } from '@geti/ui';
import { open } from '@tauri-apps/plugin-dialog';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';
import { LocalFolderSinkConfig } from '../utils';

type LocalFolderProps = {
    defaultState: LocalFolderSinkConfig;
};

const normalizeSelectedPath = (selectedPath: string | string[] | null): string | null => {
    if (typeof selectedPath === 'string') {
        return selectedPath;
    }

    if (Array.isArray(selectedPath)) {
        return selectedPath[0] ?? null;
    }

    return null;
};

export const LocalFolder = ({ defaultState }: LocalFolderProps) => {
    const [folderPath, setFolderPath] = useState(defaultState.folder_path);

    useEffect(() => {
        setFolderPath(defaultState.folder_path);
    }, [defaultState.folder_path]);

    const handleOpenFolderDialog = useCallback(() => {
        void open({
            directory: true,
            multiple: false,
            defaultPath: folderPath || undefined,
        }).then((selectedPath) => {
            const nextPath = normalizeSelectedPath(selectedPath);

            if (nextPath !== null) {
                setFolderPath(nextPath);
            }
        });
    }, [folderPath]);

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
                <div onClick={handleOpenFolderDialog} style={{ width: '100%' }}>
                    <TextField
                        isRequired
                        isReadOnly
                        width={'100%'}
                        label='Folder Path'
                        name='folder_path'
                        value={folderPath}
                    />
                </div>
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
