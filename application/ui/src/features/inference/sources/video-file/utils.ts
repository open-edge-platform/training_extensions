// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { VideoFileSourceConfig } from '../util';

export const getVideoFileInitialConfig = (): VideoFileSourceConfig => ({
    id: '',
    name: '',
    source_type: 'video_file',
    video_path: '',
});

export const videoFileBodyFormatter = (formData: FormData): VideoFileSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'video_file',
    video_path: String(formData.get('video_path')),
});
