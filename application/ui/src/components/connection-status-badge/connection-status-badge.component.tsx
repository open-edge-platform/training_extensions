// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Loading, Text } from '@geti/ui';
import { Checkmark, CloseSmall } from '@geti/ui/icons';

import classes from './connection-status-badge.module.scss';

type ConnectionStatusBadgeProps = {
    isInUse: boolean;
    isUnreachable: boolean;
    isPending: boolean;
};

export const ConnectionStatusBadge = ({ isInUse, isUnreachable, isPending }: ConnectionStatusBadgeProps) => {
    if (isPending) {
        return (
            <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
                <Loading mode='inline' size='S' />
                <Text>Checking</Text>
            </Flex>
        );
    }

    if (isInUse) {
        return (
            <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
                <Checkmark size='S' UNSAFE_className={classes.inUseIcon} />
                <Text>In use</Text>
            </Flex>
        );
    }

    if (isUnreachable) {
        return (
            <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
                <CloseSmall className={classes.unreachableIcon} />
                <Text>Unreachable</Text>
            </Flex>
        );
    }

    return (
        <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
            <Checkmark size='S' UNSAFE_className={classes.reachableIcon} />
            <Text>Reachable</Text>
        </Flex>
    );
};
