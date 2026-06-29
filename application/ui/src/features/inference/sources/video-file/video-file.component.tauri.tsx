// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react';

import { Flex, TextField } from '@geti-ui/ui';
import { open } from '@tauri-apps/plugin-dialog';

import type { VideoFileSourceConfig } from '../../../../constants/shared-types';
import { normalizeSelectedPath } from '../../shared/tauri-dialog';

type VideoFileProps = {
    defaultState?: VideoFileSourceConfig;
};

export const VideoFile = ({ defaultState }: VideoFileProps) => {
    const [videoPath, setVideoPath] = useState(defaultState?.video_path ? String(defaultState.video_path) : '');

    useEffect(() => {
        setVideoPath(defaultState?.video_path ? String(defaultState.video_path) : '');
    }, [defaultState?.video_path]);

    const handleOpenFileDialog = useCallback(() => {
        void open({
            directory: false,
            multiple: false,
            defaultPath: videoPath || undefined,
            filters: [
                {
                    name: 'Video',
                    extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v', 'mpg', 'mpeg'],
                },
            ],
        })
            .then((selectedPath) => {
                const nextPath = normalizeSelectedPath(selectedPath);

                if (nextPath !== null) {
                    setVideoPath(nextPath);
                }
            })
            .catch(console.error);
    }, [videoPath]);

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField width='100%' label='Name' name='name' defaultValue={defaultState?.name} />

            <Flex direction='row' gap='size-200'>
                <div
                    onClick={handleOpenFileDialog}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && handleOpenFileDialog()}
                    role='button'
                    tabIndex={0}
                    style={{ width: '100%', cursor: 'pointer' }}
                >
                    <TextField
                        isRequired
                        isReadOnly
                        flex='1'
                        width='100%'
                        name='video_path'
                        label='Video file path'
                        value={videoPath}
                    />
                </div>
            </Flex>
        </Flex>
    );
};
