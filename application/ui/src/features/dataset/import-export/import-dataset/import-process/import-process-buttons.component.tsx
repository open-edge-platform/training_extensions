// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';

import { $api } from '../../../../../api/client';
import { useCancelJob } from '../../../../../hooks/api/jobs.hook';
import { isInvalidStagedFile } from '../../util';

type ImportProcessButtonsProps = {
    prepareJobId: string;
    stagedDatasetId: string;
    onClose: () => void;
};

export const ImportProcessButtons = ({ prepareJobId, stagedDatasetId, onClose }: ImportProcessButtonsProps) => {
    const cancelJobMutation = useCancelJob();
    const { deleteImportEntry } = useImportDatasetToProject();
    const deleteFileMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleCancelJob = async (jobId: string) => {
        await cancelJobMutation.mutateAsync({ params: { path: { job_id: jobId } } });
        await deleteFileMutation.mutateAsync(
            { params: { path: { staged_dataset_id: stagedDatasetId } } },
            {
                onSuccess: () => {
                    deleteImportEntry(stagedDatasetId);
                    onClose();
                },
                onError: (error) => {
                    isInvalidStagedFile(error) && deleteImportEntry(stagedDatasetId);
                },
            }
        );
    };

    return (
        <ButtonGroup>
            <Button
                variant='negative'
                isPending={cancelJobMutation.isPending}
                isDisabled={cancelJobMutation.isPending}
                onPress={() => handleCancelJob(prepareJobId)}
            >
                Cancel
            </Button>
            <Button
                onPress={onClose}
                variant='secondary'
                isPending={cancelJobMutation.isPending}
                isDisabled={cancelJobMutation.isPending}
            >
                Hide
            </Button>
        </ButtonGroup>
    );
};
