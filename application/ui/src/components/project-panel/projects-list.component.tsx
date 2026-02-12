// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Project } from '../../constants/shared-types';
import { ProjectListItem } from './project-list-item/project-list-item.component';

import classes from './projects-list.module.scss';

type ProjectListProps = {
    projects: Project[];
};

export const ProjectsList = ({ projects }: ProjectListProps) => {
    return (
        <ul className={classes.projectList}>
            {projects.map((project) => (
                <ProjectListItem key={project.id} project={project} />
            ))}
        </ul>
    );
};
