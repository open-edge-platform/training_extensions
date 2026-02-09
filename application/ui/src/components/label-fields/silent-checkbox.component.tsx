// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkmark } from '@geti/ui/icons';
import { clsx } from 'clsx';

import classes from './silent-checkbox.module.scss';

type SilentCheckboxProps = {
    isSelected: boolean;
    onChange: () => void;
    'aria-label': string;
};

export const SilentCheckbox = ({ isSelected, onChange, 'aria-label': ariaLabel }: SilentCheckboxProps) => {
    return (
        <button
            className={clsx(classes.silentCheckbox, { [classes.selected]: isSelected })}
            onClick={onChange}
            aria-label={ariaLabel}
            aria-pressed={isSelected}
            type='button'
        >
            {isSelected && <Checkmark />}
        </button>
    );
};
