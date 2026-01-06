// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, View } from '@geti/ui';

import { CreateProjectForm } from '../../features/project/create/create-project-form';
import { useProjectsQuery } from '../../hooks/api/project.hook';

import backgroundStyles from '../../features/project/project-background.module.scss';

export const CreateProject = () => {
    const { data: projects = [] } = useProjectsQuery();
    const numberOfProjects = projects.length;

    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} height='100%' width='100%'>
            <Grid rows={['auto', '1fr', 'auto']} height='100%' width='100%'>
                <CreateProjectForm numberOfProjects={numberOfProjects} />
            </Grid>
        </View>
    );
};
