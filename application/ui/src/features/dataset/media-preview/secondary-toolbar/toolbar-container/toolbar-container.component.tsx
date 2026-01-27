// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti/ui';

type ToolbarContainerProps = {
    children: ReactNode;
    isHidden?: boolean;
};

export const ToolbarContainer = ({ children, isHidden }: ToolbarContainerProps) => {
    return (
        <View borderRadius={'regular'} backgroundColor={'gray-200'} padding={'size-50'} isHidden={isHidden}>
            <View backgroundColor={'gray-50'} padding={'size-100'}>
                {children}
            </View>
        </View>
    );
};
