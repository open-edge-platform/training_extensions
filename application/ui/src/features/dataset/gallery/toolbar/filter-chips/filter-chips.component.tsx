// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti-ui/ui';
import { CloseSmall } from '@geti-ui/ui/icons';

import classes from './filter-chips.module.scss';

type FilterChipsProps = {
    name: string;
    onClose: () => void;
};

export const FilterChips = ({ name, onClose }: FilterChipsProps) => {
    return (
        <Flex UNSAFE_className={classes.container} alignItems={'center'} gap={'size-75'}>
            <Text UNSAFE_className={classes.name}>{name}</Text>

            <CloseSmall
                className={classes.closeIcon}
                onClick={(event) => {
                    event.stopPropagation();
                    onClose();
                }}
            />
        </Flex>
    );
};
