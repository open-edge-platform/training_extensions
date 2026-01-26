// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjects } from './api/project.hook';
import { useProjectIdentifier } from './use-project-identifier.hook';

export const useSelectedProject = () => {
    const { data } = useProjects();
    const projectId = useProjectIdentifier();
    const selectedProject = data.find((project) => project.id === projectId);

    if (selectedProject === undefined) {
        throw new Error(`Project with id ${projectId} not found`);
    }

    return selectedProject;
};
