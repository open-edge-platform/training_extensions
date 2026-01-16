// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Button, DialogTrigger, Loading } from '@geti/ui';

import { TrainModelDialog } from './train-model-dialog.component';

export const TrainModel = () => {
    return (
        <DialogTrigger>
            <Button>Train model</Button>
            {(close) => (
                <Suspense fallback={<Loading mode={'inline'} />}>
                    <TrainModelDialog onClose={close} />
                </Suspense>
            )}
        </DialogTrigger>
    );
};
