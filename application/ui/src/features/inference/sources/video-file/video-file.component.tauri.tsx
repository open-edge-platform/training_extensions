// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react';

import { Flex, TextField } from '@geti/ui';
import { open } from '@tauri-apps/plugin-dialog';

import type { VideoFileSourceConfig } from '../../../../constants/shared-types';

type VideoFileProps = {
    defaultState?: VideoFileSourceConfig;
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
        }).then((selectedPath) => {
            const nextPath = normalizeSelectedPath(selectedPath);

            if (nextPath !== null) {
                setVideoPath(nextPath);
            }
        });
    }, [videoPath]);

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField width='100%' label='Name' name='name' defaultValue={defaultState?.name} />

            <Flex direction='row' gap='size-200'>
                <div onClick={handleOpenFileDialog} style={{ width: '100%' }}>
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
