// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react';

import { Flex, Switch, TextField } from '@geti/ui';
import { open } from '@tauri-apps/plugin-dialog';

import type { ImagesFolderSourceConfig } from '../../../../constants/shared-types';
import { normalizeSelectedPath } from '../../shared/tauri-dialog';

type ImageFolderProps = {
    defaultState?: ImagesFolderSourceConfig;
};

export const ImageFolder = ({ defaultState }: ImageFolderProps) => {
    const [imagesFolderPath, setImagesFolderPath] = useState(defaultState?.images_folder_path ?? '');

    useEffect(() => {
        setImagesFolderPath(defaultState?.images_folder_path ?? '');
    }, [defaultState?.images_folder_path]);

    const handleOpenFolderDialog = useCallback(() => {
        void open({
            directory: true,
            multiple: false,
            defaultPath: imagesFolderPath || undefined,
        })
            .then((selectedPath) => {
                const nextPath = normalizeSelectedPath(selectedPath);

                if (nextPath !== null) {
                    setImagesFolderPath(nextPath);
                }
            })
            .catch(console.error);
    }, [imagesFolderPath]);

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField width={'100%'} label='Name' name='name' defaultValue={defaultState?.name} />

            <Flex direction='row' gap='size-200'>
                <div
                    onClick={handleOpenFolderDialog}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && handleOpenFolderDialog()}
                    role='button'
                    tabIndex={0}
                    style={{ width: '100%', cursor: 'pointer' }}
                >
                    <TextField
                        isRequired
                        isReadOnly
                        flex='1'
                        width={'100%'}
                        label='Images folder path'
                        name='images_folder_path'
                        value={imagesFolderPath}
                    />
                </div>
            </Flex>

            <Switch
                aria-label='ignore existing images'
                name='ignore_existing_images'
                defaultSelected={defaultState?.ignore_existing_images}
                key={defaultState?.ignore_existing_images ? 'true' : 'false'}
            >
                Ignore existing images
            </Switch>
        </Flex>
    );
};
