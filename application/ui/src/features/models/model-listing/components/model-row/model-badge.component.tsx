// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Badge, Flex, Text } from '@geti-ui/ui';

import styles from './model-row.module.scss';

type ModelBadgeProps = {
    children: ReactNode;
    id?: string;
    color?: string;
};

export const ModelBadge = ({ children, id, color = 'var(--spectrum-global-color-gray-200)' }: ModelBadgeProps) => {
    return (
        <Badge
            data-testid={id}
            variant={'neutral'}
            UNSAFE_style={{ '--modelBadgeColor': color }}
            UNSAFE_className={styles.modelBadge}
        >
            <Text>
                <Flex alignItems={'center'} gap={'size-50'}>
                    {children}
                </Flex>
            </Text>
        </Badge>
    );
};
