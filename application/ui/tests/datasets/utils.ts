// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';

import { Job } from '../../src/constants/shared-types';
import { http } from '../fixtures';

export const STAGED_DATASET_ID = 'staged-dataset-789';
export const PREPARE_JOB_ID = 'prepare-job-123';
export const IMPORT_JOB_ID = 'import-job-456';
export const DATASET_FILENAME = 'my-dataset.zip';

export const makePrepareJob = (overrides: Partial<Job> = {}) =>
    getMockedJob({
        job_id: PREPARE_JOB_ID,
        job_type: 'prepare_dataset_for_import',
        status: 'RUNNING',
        progress: 50,
        message: 'Analyzing dataset archive...',
        ...overrides,
    });

export const makeImportJob = (
    jobType: 'import_dataset_as_new_project' | 'import_dataset_to_project',
    overrides: Partial<Job> = {}
) =>
    getMockedJob({
        job_id: IMPORT_JOB_ID,
        job_type: jobType,
        status: 'RUNNING',
        progress: 0,
        message: 'Importing dataset...',
        ...overrides,
    });

export const stagedDatasetWithMetadata = getMockedStagedDataset({
    id: STAGED_DATASET_ID,
    ready_for_import: true,
    metadata: {
        labels: ['cat', 'dog'],
        num_images: 100,
        num_annotated_images: 80,
        num_frames: 0,
        num_annotated_frames: 0,
        num_annotations: 200,
        annotation_type: 'bounding_box',
        num_videos: 0,
    },
});

export const jobPollHandler = ({
    jobId,
    whileRunning,
    whenDone,
    afterPolls = 2,
}: {
    jobId: string;
    whileRunning: ReturnType<typeof getMockedJob>;
    whenDone: ReturnType<typeof getMockedJob>;
    afterPolls?: number;
}) => {
    let pollCount = 0;

    return (requestJobId: string) => {
        if (requestJobId !== jobId) return undefined;
        pollCount += 1;
        return pollCount <= afterPolls ? whileRunning : whenDone;
    };
};

export const deleteStagedDatasetHandler = () => {
    let deletedId: string | undefined;

    const handler = http.delete('/api/staged_datasets/{staged_dataset_id}', ({ params }) => {
        deletedId = params.staged_dataset_id as string;
        return new HttpResponse(null, { status: 204 });
    });

    return { handler, getDeletedId: () => deletedId };
};
