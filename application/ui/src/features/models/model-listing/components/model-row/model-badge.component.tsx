// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Badge, Flex, Text } from '@geti/ui';

import styles from './model-row.module.scss';

type ModelBadgeProps = {
    children: ReactNode;
    id?: string;
};

export const ModelBadge = ({ children, id }: ModelBadgeProps) => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={styles.modelBadge} data-testid={id}>
            <Text>
                <Flex alignItems={'center'} gap={'size-50'}>
                    {children}
                </Flex>
            </Text>
        </Badge>
    );
};
