// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { orderBy } from 'lodash-es';

import { Project } from '../../../../constants/shared-types';
import { SortBy } from './sort-projects.component';

export const SORT_BY_HANDLERS: Record<SortBy, (projects: Project[]) => Project[]> = {
    'name-ascending': (projects) => orderBy(projects, (project) => project.name.toLocaleLowerCase(), 'asc'),
    'name-descending': (projects) => orderBy(projects, (project) => project.name.toLocaleLowerCase(), 'desc'),
    'createdAt-ascending': (projects) => orderBy(projects, (project) => project.created_at, 'asc'),
    'createdAt-descending': (projects) => orderBy(projects, (project) => project.created_at, 'desc'),
};
