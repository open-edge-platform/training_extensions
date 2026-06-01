// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export type CreateProjectInput = {
    withPrefix: string;
    task: 'classification' | 'detection' | 'instance_segmentation';
    labels: string[];
};

export type UploadMediaItem = {
    name: string;
    mimeType: string;
    buffer: Buffer;
};

export type CreatedProject = {
    id: string;
    name: string;
};

export type InferenceSourceSinkConfig = {
    sourceName: string;
    sinkName: string;
    sinkFolderPath: string;
    rateLimitSamples: number;
    rateLimitSeconds: number;
};

export type FlowInput = {
    page: Page;
    withPrefix: string;
};
