// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Button, DialogTrigger, Loading, View } from '@geti/ui';

import { TrainModelDialog } from './train-model-dialog.component';
import { TrainModelProvider } from './train-model-provider.component';

type TrainModelProps = {
    preSelectedDatasetRevisionId?: string;
    preSelectedModelRevisionId?: string;
};

export const TrainModel = ({ preSelectedDatasetRevisionId, preSelectedModelRevisionId }: TrainModelProps) => {
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
                    <TrainModelProvider
                        preSelectedDatasetRevisionId={preSelectedDatasetRevisionId}
                        preSelectedModelRevisionId={preSelectedModelRevisionId}
                    >
                        <TrainModelDialog onClose={close} />
                    </TrainModelProvider>
                </Suspense>
            )}
        </DialogTrigger>
    );
};
