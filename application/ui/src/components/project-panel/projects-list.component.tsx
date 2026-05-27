// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Project } from '../../constants/shared-types';
import { ProjectListItem } from './project-list-item/project-list-item.component';

type ProjectListProps = {
    projects: Project[];
};

export const ProjectsList = ({ projects }: ProjectListProps) => {
    const projectNames = projects.map(({ name }) => name);

    return (
        <ul>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    projectNames={projectNames.filter((name) => name !== project.name)}
                />
            ))}
        </ul>
    );
};
