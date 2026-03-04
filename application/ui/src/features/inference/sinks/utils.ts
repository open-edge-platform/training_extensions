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
const parseFiniteNumber = (value: string): number | null => {
    const compactValue = value.replace(/\s/g, '');

    if (isEmpty(compactValue)) {
        return null;
    }

    if (compactValue.includes(',') && compactValue.includes('.')) {
        return null;
    }

    const normalizedValue = compactValue.replace(',', '.');
    const parsedValue = Number(normalizedValue);

    return Number.isFinite(parsedValue) ? parsedValue : null;
};

const toDisplayNumber = (value: number) => {
    const roundedValue = Number(value.toFixed(2));

    return String(roundedValue);
};

export const positiveNumberOrUndefined = (value: number | null | undefined): number | undefined => {
    return typeof value === 'number' && value > 0 ? value : undefined;
};

export const getObjectFromFormData = (keys: FormDataEntryValue[], values: FormDataEntryValue[]) => {
    const entries = keys.map((key, index) => [key, values[index]]);
    const validEntries = entries.filter(
        ([key, value]) => !isEmpty(toStringAndTrim(key)) && !isEmpty(toStringAndTrim(value))
    );

    return Object.fromEntries(validEntries);
};

export const rateLimitFromFormData = (formData: FormData): number | null => {
    const samplesValue = formData.get('rate_limit_samples');
    const secondsValue = formData.get('rate_limit_seconds');

    if (samplesValue === null || secondsValue === null) {
        return null;
    }

    const samplesRaw = toStringAndTrim(samplesValue);
    const secondsRaw = toStringAndTrim(secondsValue);

    const samples = parseFiniteNumber(samplesRaw);
    const seconds = parseFiniteNumber(secondsRaw);

    if (samples === null || seconds === null || samples <= 0 || seconds <= 0) {
        return null;
    }

    return samples / seconds;
};

export const formatRateLimit = (rateLimit: number | null | undefined): string => {
    const normalizedRateLimit = positiveNumberOrUndefined(rateLimit);

    if (normalizedRateLimit === undefined) {
        return 'Not set';
    }

    if (normalizedRateLimit < 1) {
        const seconds = 1 / normalizedRateLimit;
        const normalizedSeconds = toDisplayNumber(seconds);
        const secondsLabel = normalizedSeconds === '1' ? 'second' : 'seconds';

        return `1 sample every ${normalizedSeconds} ${secondsLabel}`;
    }

    const normalizedSamples = toDisplayNumber(normalizedRateLimit);
    const sampleLabel = normalizedSamples === '1' ? 'sample' : 'samples';

    return `${normalizedSamples} ${sampleLabel} every 1 second`;
};
