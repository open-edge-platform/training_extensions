// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, StatusLight, Text } from '@geti/ui';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

import classes from './connection-status-badge.module.scss';

dayjs.extend(relativeTime);

type ConnectionStatusBadgeProps = {
    isAvailable: boolean;
    lastCheckedAt: number;
};

export const ConnectionStatusBadge = ({ isAvailable, lastCheckedAt }: ConnectionStatusBadgeProps) => {
    return (
        <Flex gap={'size-50'} alignItems={'center'} UNSAFE_className={classes.container}>
            <StatusLight variant={isAvailable ? 'positive' : 'negative'} UNSAFE_className={classes.statusLight}>
                {isAvailable ? 'Available |' : 'Unavailable |'}
            </StatusLight>
            <Text>Last checked: {dayjs(lastCheckedAt).fromNow()}</Text>
        </Flex>
    );
};
