// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProject } from 'hooks/api/project.hook';

import { Label } from '../types';

export const useProjectLabels = (): Label[] => {
    const { data: project } = useProject();

    return project.task.labels || [];
};
