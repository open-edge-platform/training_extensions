// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { isString } from 'lodash';

import { $api } from '../../../../../api/client';
import { useExportDataset } from '../../../../../hooks/localStorage/use-export-dataset.hook';
import { useProjectIdentifier } from '../../../../../hooks/use-project-identifier.hook';

type FormValues = {
    labels: string[];
    export_format: string;
    include_unannotated: boolean;
};

const initialState: FormValues = {
    labels: [],
    export_format: 'geti',
    include_unannotated: false,
};

type useExportDatasetJobActionProps = {
    onSuccess: () => void;
};

export const useExportDatasetJobAction = ({ onSuccess }: useExportDatasetJobActionProps) => {
    const projectId = useProjectIdentifier();
    const { addLsExportId } = useExportDataset();
    const exportJobMutation = $api.useMutation('post', '/api/jobs');

    return useActionState<FormValues, FormData>(async (_prevState, formData) => {
        const options: FormValues = {
            export_format: String(formData.get('export_format')),
            labels: formData.getAll('labels').filter(isString),
            include_unannotated: formData.get('include_unannotated') === 'on',
        };

        const { job_id } = await exportJobMutation.mutateAsync({
            body: {
                project_id: projectId,
                dataset_id: null,
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

        addLsExportId(job_id);
        onSuccess();

        return options;
    }, initialState);
};
