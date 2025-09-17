// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@geti/ui';

import { CreateProjectForm } from '../../features/project/create/create-project-form';
import Background from './../../assets/background.png';

import classes from './project.module.scss';

export const CreateProject = () => {
    return (
        <Grid
            UNSAFE_className={classes.grid}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
            }}
            rows={['auto', '1fr', 'auto']}
            height='100%'
            width='100%'
        >
            <CreateProjectForm />
        </Grid>
    );
};
