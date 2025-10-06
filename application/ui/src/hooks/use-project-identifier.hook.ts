// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useParams } from 'react-router';

export const useProjectIdentifier = () => {
    const { projectId } = useParams<{ projectId: string }>();

    if (!projectId) {
        throw new Error('No projectId found in the route');
    }

    return projectId;
};
