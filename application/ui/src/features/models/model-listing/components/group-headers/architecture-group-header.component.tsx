// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Heading, Text } from '@geti/ui';

import { type ModelArchitectureWithPerformanceCategory } from '../../../../../constants/shared-types';
import { PerformanceCategoryBadge } from '../model-row/performance-category-badge.component';

type ArchitectureGroupHeaderProps = {
    architecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ArchitectureGroupHeader = ({ architecture }: ArchitectureGroupHeaderProps) => {
    // Should never happen, but just in case
    if (architecture === undefined) {
        return <Text>Unknown</Text>;
    }

    return (
        <Flex alignItems={'center'} gap={'size-200'} marginBottom={'size-225'}>
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                {architecture.name}
            </Heading>

            {architecture.performanceCategory !== undefined && (
                <PerformanceCategoryBadge
                    id={'architecture-name'}
                    performanceCategory={architecture.performanceCategory}
                />
            )}
        </Flex>
    );
};
