// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ImagesFolderSourceConfig } from '../../../../constants/shared-types';

export const imagesFolderBodyFormatter = (formData: FormData): ImagesFolderSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'images_folder',
    images_folder_path: String(formData.get('images_folder_path')),
    ignore_existing_images: formData.get('ignore_existing_images') === 'on' ? true : false,
});
