// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';

import { type ModelArchitectureWithPerformanceCategory } from '../../../../../constants/shared-types';
import { PerformanceCategoryBadge } from './performance-category-badge.component';

import classes from './model-row.module.scss';

type ArchitectureColumnProps = {
    architecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ArchitectureColumn = ({ architecture }: ArchitectureColumnProps) => {
    // Should never happen, but just in case
    if (architecture === undefined) {
        return <Text>Unknown</Text>;
    }

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={classes.smallText}>
                {architecture.name} ({architecture.license})
            </Text>
            {architecture.performanceCategory !== undefined && (
                <PerformanceCategoryBadge
                    id={'architecture-name'}
                    performanceCategory={architecture.performanceCategory}
                />
            )}
        </Flex>
    );
};
