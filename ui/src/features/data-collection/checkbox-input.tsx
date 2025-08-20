// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { noop } from 'lodash-es';

import classes from './checkbox-input.module.scss';

type CheckboxProps = {
    name: string;
    isReadOnly?: boolean;
    isChecked?: boolean;
    onChange?: (checked: boolean) => void;
};

export const CheckboxInput = ({ name, isChecked = false, isReadOnly = false, onChange = noop }: CheckboxProps) => {
    return (
        <input
            type='checkbox'
            name={name}
            aria-label={name}
            checked={isChecked}
            readOnly={isReadOnly}
            className={classes.checkbox}
            onChange={({ target }) => onChange(target.checked)}
        />
    );
};
