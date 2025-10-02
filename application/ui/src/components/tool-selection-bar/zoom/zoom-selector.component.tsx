// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { Add, Remove } from '@geti/ui/icons';

import { IconWrapper } from '../icon-wrapper.component';

export const ZoomSelector = () => {
    return (
        <>
            <IconWrapper>
                <Add />
            </IconWrapper>

            <Flex>110%</Flex>

            <IconWrapper>
                <Remove />
            </IconWrapper>
        </>
    );
};
