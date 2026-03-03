// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { path } from 'static-path';

const root = path('/');
const projects = root.path('/projects');
const project = projects.path('/:projectId');
const inference = projects.path('/:projectId/inference');
const dataset = projects.path('/:projectId/dataset');
const datasetItem = dataset.path('/:datasetItemId');
const videoFrame = datasetItem.path('/:frameNumber');
const models = projects.path('/:projectId/models');

export const paths = {
    root,
    project: {
        index: projects,
        new: projects.path('/new'),
        details: project,
        inference,
        dataset: {
            index: dataset,
            item: {
                index: datasetItem,
                frame: videoFrame,
            },
        },
        models,
    },
};
