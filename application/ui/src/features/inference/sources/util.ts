// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../../../api/openapi-spec';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfig'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfig'];

export type SourceConfig = ImagesFolderSourceConfig | IPCameraSourceConfig;

export const getImageFolderData = <T extends { source_type: string }>(sources: T[]) => {
    return sources
        .filter(({ source_type }) => source_type === 'images_folder')
        .at(0) as unknown as ImagesFolderSourceConfig;
};

export const getIpCameraData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'ip_camera').at(0) as unknown as IPCameraSourceConfig;
};
