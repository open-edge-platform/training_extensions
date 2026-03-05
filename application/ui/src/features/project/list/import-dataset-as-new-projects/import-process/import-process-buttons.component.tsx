// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useCancelJob } from 'hooks/api/jobs/jobs.hook';

import { useDeleteStagedDataset } from '../../../../../hooks/api/staged-file.hook';

type ImportProcessButtonsProps = {
    prepareJobId: string;
    stagedDatasetId: string;
    onClose: () => void;
};

export const ImportProcessButtons = ({ prepareJobId, stagedDatasetId, onClose }: ImportProcessButtonsProps) => {
    const cancelJobMutation = useCancelJob();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose });

    const handleCancelJob = async (jobId: string) => {
        await cancelJobMutation.mutateAsync({ params: { path: { job_id: jobId } } });
        deleteFileMutation.mutate();
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
