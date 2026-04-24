// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../api/client';

// Use plain string concatenation instead of `new URL(path, base)`. The latter
// requires an absolute base, but `API_BASE_URL` is an empty string when the UI
// is served from the same origin as the backend (the default in Docker).
const apiPath = (path: string) => `${API_BASE_URL.replace(/\/$/, '')}/api/${path}`;

export const getProjectThumbnailUrl = (projectId: string) => apiPath(`projects/${projectId}/thumbnail`);

const getMediaBaseUrl = (projectId: string, itemId: string) => apiPath(`projects/${projectId}/dataset/media/${itemId}`);

const getDatasetRevisionItemBaseUrl = (projectId: string, datasetRevisionId: string, itemId: string) =>
    apiPath(`projects/${projectId}/dataset_revisions/${datasetRevisionId}/items/${itemId}`);

export const getThumbnailUrl = (projectId: string, itemId: string) => `${getMediaBaseUrl(projectId, itemId)}/thumbnail`;

export const getMediaBinaryUrl = (projectId: string, itemId: string) => `${getMediaBaseUrl(projectId, itemId)}/binary`;

export const getDatasetRevisionThumbnailUrl = (projectId: string, datasetRevisionId: string, itemId: string) =>
    `${getDatasetRevisionItemBaseUrl(projectId, datasetRevisionId, itemId)}/thumbnail`;

export const getVideoFrameBinaryUrl = (projectId: string, itemId: string, frameNumber: number) =>
    `${getMediaBinaryUrl(projectId, itemId)}?frame_index=${frameNumber}`;

export const getVideoFrameThumbnailUrl = (projectId: string, itemId: string, frameNumber: number) =>
    `${getMediaBaseUrl(projectId, itemId)}/thumbnail?frame_index=${frameNumber}`;
