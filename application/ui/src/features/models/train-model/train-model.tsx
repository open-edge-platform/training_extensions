// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button } from '@geti/ui';

import { TrainModelDialog } from './train-model-dialog.component';

export const TrainModel = () => {
    const [isTrainingDialogOpen, setIsTrainingDialogOpen] = useState<boolean>(false);

    return (
        <>
            <Button
                id={'train-new-model-button-id'}
                data-testid={'train-new-model-button-id'}
                variant={'accent'}
                onPress={() => setIsTrainingDialogOpen(true)}
            >
                Train model
            </Button>
            <TrainModelDialog isOpen={isTrainingDialogOpen} onClose={() => setIsTrainingDialogOpen(false)} />
        </>
    );
};
