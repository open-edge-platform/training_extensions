// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ExportDatasetJob, Job, PrepareImportDatasetJob } from '../src/constants/shared-types';

export const getMockedJob = (job?: Partial<Job>): Job => {
    return {
        job_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        job_type: 'train',
        metadata: {
            project: {
                id: '7b073838-99d3-42ff-9018-4e901eb047fc',
            },
            model: {
                id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                name: 'ATSS (ef3983f1)',
                architecture: 'Custom_Object_Detection_Gen3_ATSS',
                parent_revision_id: null,
                dataset_revision_id: '6f9f9g61-4fg1-7781-e082-e1113f371e01',
            },
            device: {
                type: 'cpu',
                name: 'CPU',
            },
        },
        status: 'RUNNING',
        progress: 45,
        message: 'Training in progress...',
        error: null,
        started_at: '2026-01-19T08:15:00.000000+00:00',
        finished_at: null,
        ...job,
    };
};

export const getMockedJobExportJob = (job: Partial<ExportDatasetJob>): ExportDatasetJob => ({
    ...getMockedJob(),
    type: 'export_dataset',
    metadata: {
        dataset_id: 'staged-dataset-123',
        project_id: 'project-123',
        filters: {
            include_unannotated: false,
        },
    },
    ...job,
});

export const getMockedPrepareImportDatasetJob = (job: Partial<PrepareImportDatasetJob>): PrepareImportDatasetJob => ({
    ...getMockedJob(),
    type: 'prepare_dataset_for_import',
    metadata: {
        staged_dataset_id: 'staged-dataset-123',
        job_type: 'prepare_dataset_for_import',
        project_id: 'project-123',
        filters: {
            include_unannotated: false,
        },
    },
    ...job,
});
