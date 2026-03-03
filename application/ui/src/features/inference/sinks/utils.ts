// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import type { components } from '../../../api/openapi-spec';

export type LocalFolderSinkConfig = components['schemas']['FolderSinkConfigView'];
export type MqttSinkConfig = components['schemas']['MqttSinkConfigView'];
export type WebhookSinkConfig = components['schemas']['WebhookSinkConfigView'];
export type DisconnectedSinkConfig = components['schemas']['RosSinkConfigView'];
export type RosSinkConfig = components['schemas']['RosSinkConfigView'];
export type SinkOutputFormats = LocalFolderSinkConfig['output_formats'];

export type SinkConfig =
    | LocalFolderSinkConfig
    | MqttSinkConfig
    | WebhookSinkConfig
    | DisconnectedSinkConfig
    | RosSinkConfig;

export enum SinkType {
    FOLDER = 'folder',
    MQTT = 'mqtt',
    ROS = 'ros',
    WEBHOOK = 'webhook',
}

export enum OutputFormat {
    IMAGE_ORIGINAL = 'image_original',
    IMAGE_WITH_PREDICTIONS = 'image_with_predictions',
    PREDICTIONS = 'predictions',
}

export enum WebhookHttpMethod {
    PUT = 'PUT',
    POST = 'POST',
    PATCH = 'PATCH',
}

const toStringAndTrim = (value: unknown) => String(value).trim();

export const getObjectFromFormData = (keys: FormDataEntryValue[], values: FormDataEntryValue[]) => {
    const entries = keys.map((key, index) => [key, values[index]]);
    const validEntries = entries.filter(
        ([key, value]) => !isEmpty(toStringAndTrim(key)) && !isEmpty(toStringAndTrim(value))
    );

    return Object.fromEntries(validEntries);
};
