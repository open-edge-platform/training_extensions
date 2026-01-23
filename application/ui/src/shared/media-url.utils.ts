// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../api/client';

const getMediaBaseUrl = (projectId: string, itemId: string) =>
    `${API_BASE_URL}/api/projects/${projectId}/dataset/media/${itemId}`;

export const getThumbnailUrl = (projectId: string, itemId: string) => {
    return `${getMediaBaseUrl(projectId, itemId)}/thumbnail`;
};

export const getMediaBinaryUrl = (projectId: string, itemId: string) => {
    return `${getMediaBaseUrl(projectId, itemId)}/binary`;
};
