// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps } from 'react';

import { View } from '@geti-ui/ui';

type ViewProps = ComponentProps<typeof View>;

type ToolbarContainerProps = Omit<ViewProps, 'borderRadius' | 'backgroundColor' | 'padding'>;

const ToolbarContainer = ({ children, ...rest }: ToolbarContainerProps) => {
    return (
        <View borderRadius={'regular'} backgroundColor={'gray-200'} padding={'size-50'} {...rest}>
            {children}
        </View>
    );
};

type ToolbarSectionProps = Omit<ViewProps, 'borderRadius' | 'backgroundColor' | 'padding'>;

const ToolbarSection = ({ children, ...rest }: ToolbarSectionProps) => {
    return (
        <View borderRadius={'regular'} backgroundColor={'gray-50'} padding={'size-100'} {...rest}>
            {children}
        </View>
    );
};

export const Toolbar = {
    Container: ToolbarContainer,
    Section: ToolbarSection,
};
