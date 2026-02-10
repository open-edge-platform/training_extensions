// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { useLocalStorageDataset } from 'hooks/use-local-storage-dataset.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isString } from 'lodash';

import { $api } from '../../../../api/client';
import { components } from '../../../../api/openapi-spec';

type DatasetItemSubset = components['schemas']['DatasetItemSubset'];

type FormValues = {
    labels: string[];
    include_unannotated: boolean;
    subsets: DatasetItemSubset[];
};

const initialState: FormValues = {
    labels: [],
    include_unannotated: false,
    subsets: [],
};

type useExportDatasetJobActionProps = {
    onSuccess: () => void;
};

export const useExportDatasetJobAction = ({ onSuccess }: useExportDatasetJobActionProps) => {
    const project_id = useProjectIdentifier();
    const { addLsExportId } = useLocalStorageDataset();
    const exportJobMutation = $api.useMutation('post', '/api/jobs');

    return useActionState<FormValues, FormData>(async (_prevState, formData) => {
        const filters: FormValues = {
            labels: formData.getAll('labels').filter(isString),
            subsets: formData.getAll('subsets') as DatasetItemSubset[],
            include_unannotated: formData.get('include_unannotated') === 'on',
        };

        const response = await exportJobMutation.mutateAsync({
            body: {
                project_id,
                job_type: 'export_dataset',
                parameters: { filters, export_format: 'coco' },
            },
        });

        addLsExportId(response.job_id);
        onSuccess();

        return filters;
    }, initialState);
};
