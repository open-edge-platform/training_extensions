// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { ModelBadge } from './model-badge.component';

import styles from './model-row.module.scss';

type ArchitectureColumnProps = {
    architecture: string;
};

export const ArchitectureColumn = ({ architecture }: ArchitectureColumnProps) => {
    return (
        <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={styles.smallText}>{architecture} (Apache 2.0)</Text>
            {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
            <ModelBadge id={'architecture-name'}>
                <ThumbsUp />
                <Text>Speed</Text>
            </ModelBadge>
        </Flex>
    );
};
