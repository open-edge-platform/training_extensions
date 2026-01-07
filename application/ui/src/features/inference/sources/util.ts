// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../../../api/openapi-spec';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfigView'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfigView'];
export type USBCameraSourceConfig = components['schemas']['USBCameraSourceConfigView'];
export type VideoFileSourceConfig = components['schemas']['VideoFileSourceConfigView'];

export type SourceConfig =
    | ImagesFolderSourceConfig
    | IPCameraSourceConfig
    | USBCameraSourceConfig
    | VideoFileSourceConfig;

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
    return sources.filter(({ source_type }) => source_type === 'usb_camera').at(0) as unknown as USBCameraSourceConfig;
};

export const getVideoFileData = <T extends { source_type: string }>(sources: T[]) => {
    return sources.filter(({ source_type }) => source_type === 'video_file').at(0) as unknown as VideoFileSourceConfig;
};
