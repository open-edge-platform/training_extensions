// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti/ui';

type ToolbarContainerProps = {
    children: ReactNode;
    isHidden?: boolean;
};

const ToolbarContainer = ({ children, isHidden }: ToolbarContainerProps) => {
    return (
        <View borderRadius={'regular'} backgroundColor={'gray-200'} padding={'size-50'} isHidden={isHidden}>
            {children}
        </View>
    );
};

type ToolbarSectionProps = {
    children: ReactNode;
};

const ToolbarSection = ({ children }: ToolbarSectionProps) => {
    return (
        <View backgroundColor={'gray-50'} padding={'size-100'}>
            {children}
        </View>
    );
};

export const Toolbar = {
    Container: ToolbarContainer,
    Section: ToolbarSection,
};
