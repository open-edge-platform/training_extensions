// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti/ui';
import { CloseSmall } from '@geti/ui/icons';

import classes from './filter-chips.module.scss';

type FilterChipsProps = {
    name: string;
    onClose: () => void;
};

export const FilterChips = ({ name, onClose }: FilterChipsProps) => {
    return (
        <div className={classes.container}>
            <Text UNSAFE_className={classes.name}>{name}</Text>

            <CloseSmall
                className={classes.closeIcon}
                onClick={(event) => {
                    event.stopPropagation();
                    onClose();
                }}
            />
        </div>
    );
};
