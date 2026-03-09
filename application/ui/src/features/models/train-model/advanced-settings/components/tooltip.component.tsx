// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, ContextualHelp, Text } from '@geti/ui';

export const Tooltip = ({ children }: { children: ReactNode }) => {
    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{children}</Text>
            </Content>
        </ContextualHelp>
    );
};
