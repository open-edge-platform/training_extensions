// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Button, DialogTrigger, Loading, View } from '@geti/ui';

import { usePrefetchTrainModelData } from './api/use-prefetch-train-model-data';
import { TrainModelDialog } from './train-model-dialog.component';
import { TrainModelProvider } from './train-model-provider.component';

export const TrainModel = () => {
    usePrefetchTrainModelData();

    return (
        <DialogTrigger>
            <Button margin={0}>Train model</Button>
            {(close) => (
                <Suspense
                    fallback={
                        <View padding={'size-2400'}>
                            <Loading mode={'inline'} />
                        </View>
                    }
                >
                    <TrainModelProvider>
                        <TrainModelDialog onClose={close} />
                    </TrainModelProvider>
                </Suspense>
            )}
        </DialogTrigger>
    );
};
