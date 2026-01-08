// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { USBCameraSourceConfig } from '../util';

export const getUsbCameraInitialConfig = (): USBCameraSourceConfig => ({
    id: '',
    name: '',
    source_type: 'usb_camera',
    device_id: 0,
});

export const usbCameraBodyFormatter = (formData: FormData): USBCameraSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'usb_camera',
    device_id: Number(formData.get('device_id')),
});
