// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti-ui/ui';
import { capitalize } from 'lodash-es';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { ModelBadge } from './model-badge.component';

type PerformanceCategoryBadgeProps = {
    performanceCategory: string;
    id?: string;
    color?: string;
};

export const PerformanceCategoryBadge = ({ performanceCategory, id, color }: PerformanceCategoryBadgeProps) => {
    return (
        <ModelBadge id={id} color={color}>
            <ThumbsUp />
            <Text>{capitalize(performanceCategory)}</Text>
        </ModelBadge>
    );
};
