// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { useNavigate } from 'react-router';

import type { SchemaProjectView } from '../../api/openapi-spec';
import { paths } from '../../constants/paths';
import { useDeleteProject, usePatchProject } from '../../hooks/api/project.hook';
import { ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: SchemaProjectView[];
    projectIdInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectsList = ({ projects, setProjectInEdition, projectIdInEdition }: ProjectListProps) => {
    const deleteProjectMutation = useDeleteProject();
    const patchProjectMutation = usePatchProject();
    const navigate = useNavigate();
    const projectIdentifier = useProjectIdentifier();

    const updateProjectName = (id: string, name: string): void => {
        patchProjectMutation.mutate(
            {
                params: { path: { project_id: id } },
                body: { name },
            },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: 'Project updated successfully' });
                },
            }
        );
    };

    const deleteProject = (id: string): void => {
        deleteProjectMutation.mutate(
            {
                params: {
                    path: {
                        project_id: id,
                    },
                },
            },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: 'Project deleted successfully' });

                    if (projects.length === 1) {
                        navigate(paths.project.new({}));
                    } else if (id === projectIdentifier) {
                        navigate(paths.project.index({}));
                    }
                },
            }
        );
    };

    const isInEditionMode = (projectId: string) => {
        return projectIdInEdition === projectId;
    };

    const handleBlur = (projectId: string, newName: string) => {
        setProjectInEdition(null);

        const projectToUpdate = projects.find((project) => project.id === projectId);
        if (projectToUpdate?.name === newName || isEmpty(newName.trim())) {
            return;
        }

        updateProjectName(projectId, newName);
    };

    const handleRename = (projectId: string) => {
        setProjectInEdition(projectId);
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    onRename={handleRename}
                    onDelete={deleteProject}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                />
            ))}
        </ul>
    );
};
