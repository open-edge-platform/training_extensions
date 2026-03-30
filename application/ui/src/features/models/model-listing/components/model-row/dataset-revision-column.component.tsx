// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti-ui/ui';
import { Image, Tag } from '@geti-ui/ui/icons';
import { useNumberFormatter } from 'react-aria';

import type { DatasetRevision } from '../../../../../constants/shared-types';
import { formatDatasetRevisionDate } from '../../utils/date-formatting';
import { ModelBadge } from './model-badge.component';

import styles from './model-row.module.scss';

type DatasetColumnProps = {
    datasetRevision: DatasetRevision | undefined;
    labelsCount: number | undefined;
};

export const DatasetColumn = ({ datasetRevision, labelsCount }: DatasetColumnProps) => {
    const totalCount = datasetRevision?.item_counts?.total;
    const formatter = useNumberFormatter();

    // Should never happen, but just in case
    if (datasetRevision === undefined) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'}>
                Unknown
            </Flex>
        );
    }

    return (
        <Flex direction={'column'} gap={'size-50'}>
            <Text UNSAFE_className={styles.datasetRevisionName}>{datasetRevision.name}</Text>
            <Text UNSAFE_className={styles.datasetRevisionDate}>
                {formatDatasetRevisionDate(datasetRevision.created_at)}
            </Text>
            <Flex gap={'size-100'}>
                {labelsCount !== undefined && (
                    <ModelBadge id={'labels-count'}>
                        <Tag />
                        <Text>{labelsCount}</Text>
                    </ModelBadge>
                )}
                {totalCount !== undefined && (
                    <ModelBadge id={'dataset-count'}>
                        <Image />
                        <Text>{formatter.format(totalCount)}</Text>
                    </ModelBadge>
                )}
            </Flex>
        </Flex>
    );
};
