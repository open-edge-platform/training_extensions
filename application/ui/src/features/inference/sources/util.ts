// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../../../api/openapi-spec';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfig'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfig'];
export type WebcamSourceConfig = components['schemas']['WebcamSourceConfig'];

export type SourceConfig = ImagesFolderSourceConfig | IPCameraSourceConfig | WebcamSourceConfig;

export const isOnlyDigits = (str: string): boolean => {
    return /^\d+$/.test(str);
};

export const getImageFolderData = <T extends { source_type: string }>(sources: T[]) => {
    return sources
        .filter(({ source_type }) => source_type === 'images_folder')
        .at(0) as unknown as ImagesFolderSourceConfig;
};

export const getIpCameraData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'ip_camera').at(0) as unknown as IPCameraSourceConfig;
};

export const getWebcamData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'webcam').at(0) as unknown as WebcamSourceConfig;
};
