// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';

import { useCancelJob } from '../../../../../hooks/api/jobs.hook';

type ImportProcessButtonsProps = {
    prepareJobId: string;
    onClose: () => void;
};

export const ImportProcessButtons = ({ prepareJobId, onClose }: ImportProcessButtonsProps) => {
    const cancelJobMutation = useCancelJob();

    const handleCancelJob = (jobId: string) => {
        cancelJobMutation.mutate({ params: { path: { job_id: jobId } } }, { onSuccess: onClose });
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
