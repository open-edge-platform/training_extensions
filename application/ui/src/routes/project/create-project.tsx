// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { CreateProjectForm } from '../../features/project/create/create-project-form';
import { useProjects } from '../../hooks/api/project.hook';

import backgroundStyles from '../../features/project/project-background.module.scss';

export const CreateProject = () => {
    const { data: projects = [] } = useProjects();

    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} height='100%' width='100%'>
            <CreateProjectForm projects={projects} />
        </View>
    );
};
