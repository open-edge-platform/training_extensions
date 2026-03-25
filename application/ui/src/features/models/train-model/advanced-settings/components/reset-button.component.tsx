// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps } from 'react';

import { ActionButton } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

type ResetButtonProps = Omit<ComponentProps<typeof ActionButton>, 'children' | 'isQuiet'>;

export const ResetButton = (props: ResetButtonProps) => {
    return (
        <ActionButton isQuiet {...props}>
            <Refresh />
        </ActionButton>
    );
};
