// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { capitalize } from 'lodash-es';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { type ModelArchitectureWithPerformanceCategory } from '../../../../../constants/shared-types';
import { ModelBadge } from './model-badge.component';

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
            <Text UNSAFE_className={classes.smallText}>{architecture.name} (Apache 2.0)</Text>
            {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
            <ModelBadge id={'architecture-name'}>
                <ThumbsUp />
                <Text>{capitalize(architecture.performanceCategory)}</Text>
            </ModelBadge>
        </Flex>
    );
};
