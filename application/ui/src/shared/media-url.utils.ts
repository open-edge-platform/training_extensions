// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../api/client';

const getDatasetItemBaseUrl = (projectId: string, itemId: string) =>
    `${API_BASE_URL}/api/projects/${projectId}/dataset/items/${itemId}`;

export const getThumbnailUrl = (projectId: string, itemId: string) => {
    return `${getDatasetItemBaseUrl(projectId, itemId)}/thumbnail`;
};

export const getMediaBinaryUrl = (projectId: string, itemId: string) => {
    return `${getDatasetItemBaseUrl(projectId, itemId)}/binary`;
};
