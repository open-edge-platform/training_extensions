// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

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
    POST = 'POST',
    PUT = 'PUT',
}

const toStringAndTrim = (value: unknown) => String(value).trim();

export const getObjectFromFormData = (keys: FormDataEntryValue[], values: FormDataEntryValue[]) => {
    const entries = keys.map((key, index) => [key, values[index]]);
    const validEntries = entries.filter(
        ([key, value]) => !isEmpty(toStringAndTrim(key)) && !isEmpty(toStringAndTrim(value))
    );

    return Object.fromEntries(validEntries);
};
