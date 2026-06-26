// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti-ui/ui';

type TrainModelDialogLayoutProps = {
    children: ReactNode;
};

export const TrainModelDialogLayout = ({ children }: TrainModelDialogLayoutProps) => {
    return (
        <View padding={'size-300'} backgroundColor={'gray-50'} height={'100%'}>
            {children}
        </View>
    );
};
