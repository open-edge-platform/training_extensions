// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useCancelJob } from 'hooks/api/jobs/jobs.hook';
import { useDeleteStagedDataset } from 'hooks/api/use-delete-staged-dataset.hook';

type ImportProcessButtonsProps = {
    prepareJobId: string;
    stagedDatasetId: string;
    onClose: () => void;
    deleteEntry: () => void;
};

export const ImportProcessButtons = ({
    prepareJobId,
    stagedDatasetId,
    onClose,
    deleteEntry,
}: ImportProcessButtonsProps) => {
    const cancelJobMutation = useCancelJob();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose, deleteEntry });

    const isPending = cancelJobMutation.isPending || deleteFileMutation.isPending;

    const handleCancelJob = async (jobId: string) => {
        await cancelJobMutation.mutateAsync({ params: { path: { job_id: jobId } } });
        deleteFileMutation.mutate();
    };

    return (
        <ButtonGroup>
            <Button
                variant='negative'
                isPending={isPending}
                isDisabled={isPending}
                onPress={() => handleCancelJob(prepareJobId)}
            >
                Cancel
            </Button>
            <Button onPress={onClose} variant='secondary' isPending={isPending} isDisabled={isPending}>
                Hide
            </Button>
        </ButtonGroup>
    );
};
