// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../api/client';

export const getProjectThumbnailUrl = (projectId: string) => `${API_BASE_URL}/api/projects/${projectId}/thumbnail`;

const getMediaBaseUrl = (projectId: string, itemId: string) =>
    `${API_BASE_URL}/api/projects/${projectId}/dataset/media/${itemId}`;

const getDatasetRevisionItemBaseUrl = (projectId: string, datasetRevisionId: string, itemId: string) =>
    `${API_BASE_URL}/api/projects/${projectId}/dataset_revisions/${datasetRevisionId}/items/${itemId}`;

export const getThumbnailUrl = (projectId: string, itemId: string) => {
    return `${getMediaBaseUrl(projectId, itemId)}/thumbnail`;
};

export const getMediaBinaryUrl = (projectId: string, itemId: string) => {
    return `${getMediaBaseUrl(projectId, itemId)}/binary`;
};

export const getDatasetRevisionThumbnailUrl = (projectId: string, datasetRevisionId: string, itemId: string) => {
    return `${getDatasetRevisionItemBaseUrl(projectId, datasetRevisionId, itemId)}/thumbnail`;
};

export const getDatasetRevisionMediaBinaryUrl = (projectId: string, datasetRevisionId: string, itemId: string) => {
    return `${getDatasetRevisionItemBaseUrl(projectId, datasetRevisionId, itemId)}/binary`;
};

export const getVideoFrameBinaryUrl = (projectId: string, itemId: string, frameNumber: number) => {
    return `${getMediaBaseUrl(projectId, itemId)}/${frameNumber}/binary`;
};

export const getVideoFrameBinaryThumbnail = (projectId: string, itemId: string, frameNumber: number) => {
    return `${getMediaBaseUrl(projectId, itemId)}/${frameNumber}/thumbnail`;
};
