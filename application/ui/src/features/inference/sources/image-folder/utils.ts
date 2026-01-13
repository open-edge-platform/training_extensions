// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ImagesFolderSourceConfig } from '../util';

export const getImagesFolderInitialConfig = (): ImagesFolderSourceConfig => ({
    id: '',
    name: '',
    source_type: 'images_folder',
    images_folder_path: '',
    ignore_existing_images: false,
});

export const imagesFolderBodyFormatter = (formData: FormData): ImagesFolderSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'images_folder',
    images_folder_path: String(formData.get('images_folder_path')),
    ignore_existing_images: formData.get('ignore_existing_images') === 'on' ? true : false,
});
