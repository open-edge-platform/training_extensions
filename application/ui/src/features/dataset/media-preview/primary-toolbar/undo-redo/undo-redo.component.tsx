// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Redo, Undo } from '@geti/ui/icons';

import { IconWrapper } from '../../../../../components/icon-wrapper/icon-wrapper.component';

export const UndoRedo = () => {
    return (
        <>
            <IconWrapper>
                <Undo />
            </IconWrapper>

            <IconWrapper>
                <Redo />
            </IconWrapper>
        </>
    );
};
