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

    const samples = Number(samplesValue);
    const seconds = Number(secondsValue);

    if (!Number.isFinite(samples) || !Number.isFinite(seconds) || samples <= 0 || seconds <= 0) {
        return null;
    }

    return samples / seconds;
};

export const formatRateLimit = (rateLimit?: number | null): string => {
    const normalizedRateLimit = positiveNumberOrUndefined(rateLimit);

    if (normalizedRateLimit === undefined) {
        return 'Not set';
    }

    if (normalizedRateLimit < 1) {
        const seconds = 1 / normalizedRateLimit;
        const normalizedSeconds = Math.round(seconds);
        const secondsLabel = normalizedSeconds === 1 ? 'second' : 'seconds';

        return `1 sample every ${normalizedSeconds} ${secondsLabel}`;
    }

    const normalizedSamples = Math.round(normalizedRateLimit);
    const sampleLabel = normalizedSamples === 1 ? 'sample' : 'samples';

    return `${normalizedSamples} ${sampleLabel} every 1 second`;
};
