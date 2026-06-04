// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { isString } from 'lodash-es';

import { useSubmitJob } from './api/jobs/jobs.hook';
import { useExportDataset } from './storage/use-export-dataset.hook';
import { useProjectIdentifier } from './use-project-identifier.hook';

type FormValues = {
    labels: string[];
    export_format: string;
    dataset_id: string | null;
    include_unannotated: boolean;
};

const initialState: FormValues = {
    labels: [],
    dataset_id: null,
    export_format: 'geti',
    include_unannotated: true,
};

type useExportDatasetJobActionProps = {
    datasetId: string | null;
    onSuccess: () => void;
};

export const useExportDatasetJobAction = ({ datasetId, onSuccess }: useExportDatasetJobActionProps) => {
    const projectId = useProjectIdentifier();
    const exportJobMutation = useSubmitJob();
    const { addLsExportId } = useExportDataset();

    return useActionState<FormValues, FormData>(async (_prevState, formData) => {
        const options: FormValues = {
            labels: formData.getAll('labels').filter(isString),
            export_format: String(formData.get('export_format')),
            include_unannotated: formData.get('include_unannotated') === 'on',
            dataset_id: datasetId,
        };

        const { job_id } = await exportJobMutation.mutateAsync({
            body: {
                project_id: projectId,
                dataset_id: options.dataset_id,
                job_type: 'export_dataset',
                parameters: {
                    export_format: options.export_format,
                    filters: {
                        labels: options.labels,
                        include_unannotated: options.include_unannotated,
                    },
                },
            },
        });

        addLsExportId(job_id, datasetId);
        onSuccess();

        return options;
    }, initialState);
};
