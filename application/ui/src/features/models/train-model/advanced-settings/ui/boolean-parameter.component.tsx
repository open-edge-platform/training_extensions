// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Switch } from '@geti/ui';

type BooleanParameterProps = {
    value: boolean;
    header: string;
    onChange: (isSelected: boolean) => void;
    isDisabled?: boolean;
};

export const BooleanParameter = ({ value, header, onChange, isDisabled }: BooleanParameterProps) => {
    return (
        <Switch
            isEmphasized
            isSelected={value}
            aria-label={`Toggle ${header}`}
            onChange={onChange}
            isDisabled={isDisabled}
        >
            {value ? 'On' : 'Off'}
        </Switch>
    );
};
