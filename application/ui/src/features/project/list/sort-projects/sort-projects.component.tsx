// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker, Section } from '@geti/ui';
import { orderBy } from 'lodash-es';

import { Project } from '../../../../constants/shared-types';

import classes from './sort-projects.module.scss';

const SORT_BY_OPTIONS = [
    [
        { name: 'Name (A-Z)', key: 'name-ascending' },
        { name: 'Name (Z-A)', key: 'name-descending' },
    ],
    [
        { name: 'Created date (newest)', key: 'createdAt-descending' },
        { name: 'Created date (oldest)', key: 'createdAt-ascending' },
    ],
] as const;

export type SortBy = (typeof SORT_BY_OPTIONS)[number][number]['key'];

type SortProjectsProps = {
    sortBy: SortBy;
    onSort: (sortBy: SortBy) => void;
};

export const SORT_BY_HANDLERS: Record<SortBy, (projects: Project[]) => Project[]> = {
    'name-ascending': (projects) => orderBy(projects, (project) => project.name, 'asc'),
    'name-descending': (projects) => orderBy(projects, (project) => project.name, 'desc'),
    'createdAt-ascending': (projects) => orderBy(projects, (project) => project.created_at, 'asc'),
    'createdAt-descending': (projects) => orderBy(projects, (project) => project.created_at, 'desc'),
};

export const SortProjects = ({ sortBy, onSort }: SortProjectsProps) => {
    return (
        <Picker
            isQuiet
            items={SORT_BY_OPTIONS}
            selectedKey={sortBy}
            onSelectionChange={(key) => onSort(key as SortBy)}
            labelAlign={'start'}
            labelPosition={'side'}
            label={'Sort:'}
            UNSAFE_className={classes.sortProjects}
        >
            {(item) => {
                return (
                    <Section key={item.map((option) => option.key).join('-')}>
                        {item.map((option) => (
                            <Item key={option.key} textValue={option.name}>
                                {option.name}
                            </Item>
                        ))}
                    </Section>
                );
            }}
        </Picker>
    );
};
