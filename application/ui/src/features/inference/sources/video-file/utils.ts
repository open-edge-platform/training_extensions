// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { VideoFileSourceConfig } from '../../../../constants/shared-types';
import { getUniqueName } from '../utils';

export const getVideoFileInitialConfig = (existingNames: string[] = []): VideoFileSourceConfig => ({
    id: '',
    name: getUniqueName('Video file source', existingNames),
    source_type: 'video_file',
    video_path: '',
});

export const videoFileBodyFormatter = (formData: FormData): VideoFileSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'video_file',
    video_path: String(formData.get('video_path')),
});
