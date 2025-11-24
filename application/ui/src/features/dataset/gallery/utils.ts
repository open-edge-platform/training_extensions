// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '../../../api/client';

const getBaseUrl = (projectId: string, itemId: string) =>
    `${API_BASE_URL}/api/projects/${projectId}/dataset/items/${itemId}`;

export const getThumbnailUrl = (projectId: string, itemId: string) => {
    return `${getBaseUrl(projectId, itemId)}/thumbnail`;
};
