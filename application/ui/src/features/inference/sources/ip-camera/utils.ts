// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IPCameraSourceConfig } from '../util';

export const getIpCameraInitialConfig = (): IPCameraSourceConfig => ({
    id: '',
    name: '',
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
});

export const ipCameraBodyFormatter = (formData: FormData): IPCameraSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'ip_camera',
    stream_url: String(formData.get('stream_url')),
    auth_required: String(formData.get('auth_required')) === 'on' ? true : false,
});
