// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { Link } from 'react-router-dom';

import { paths } from '../../../router';

import classes from './new-project-link.module.scss';

export const NewProjectLink = () => {
    return (
        <Link to={paths.pipeline.new.pattern} className={classes.link}>
            <AddCircle />
            <Text>Add another project</Text>
        </Link>
    );
};
