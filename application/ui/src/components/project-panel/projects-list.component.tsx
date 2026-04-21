// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Project } from '../../constants/shared-types';
import { ProjectListItem } from './project-list-item/project-list-item.component';

type ProjectListProps = {
    projects: Project[];
};

export const ProjectsList = ({ projects }: ProjectListProps) => {
    return (
        <ul>
            {projects.map((project) => (
                <ProjectListItem key={project.id} project={project} />
            ))}
        </ul>
    );
};
