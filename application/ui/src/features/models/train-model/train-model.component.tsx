// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, DialogTrigger } from '@geti/ui';

import { TrainModelDialog } from './train-model-dialog.component';

export const TrainModel = () => {
    return (
        <DialogTrigger>
            <Button>Train model</Button>
            {(close) => <TrainModelDialog onClose={close} />}
        </DialogTrigger>
    );
};
