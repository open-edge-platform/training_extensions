// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../api/client';

export const getProjectThumbnailUrl = (projectId: string) =>
    new URL(`api/projects/${projectId}/thumbnail`, API_BASE_URL).toString();

const getMediaBaseUrl = (projectId: string, itemId: string) =>
    new URL(`api/projects/${projectId}/dataset/media/${itemId}`, API_BASE_URL).toString();

const getDatasetRevisionItemBaseUrl = (projectId: string, datasetRevisionId: string, itemId: string) =>
    new URL(`api/projects/${projectId}/dataset_revisions/${datasetRevisionId}/items/${itemId}`, API_BASE_URL);

export const getThumbnailUrl = (projectId: string, itemId: string) => {
    return new URL('thumbnail', `${getMediaBaseUrl(projectId, itemId)}/`).toString();
};

export const getMediaBinaryUrl = (projectId: string, itemId: string) => {
    return new URL('binary', `${getMediaBaseUrl(projectId, itemId)}/`).toString();
};

export const getDatasetRevisionThumbnailUrl = (projectId: string, datasetRevisionId: string, itemId: string) => {
    return new URL('thumbnail', `${getDatasetRevisionItemBaseUrl(projectId, datasetRevisionId, itemId)}/`).toString();
};

export const getVideoFrameBinaryUrl = (projectId: string, itemId: string, frameNumber: number) => {
    const url = new URL(getMediaBinaryUrl(projectId, itemId));

    url.searchParams.set('frame_index', frameNumber.toString());

    return url.toString();
};

export const getVideoFrameThumbnailUrl = (projectId: string, itemId: string, frameNumber: number) => {
    const url = new URL('thumbnail', `${getMediaBaseUrl(projectId, itemId)}/`);

    url.searchParams.set('frame_index', frameNumber.toString());

    return url.toString();
};
