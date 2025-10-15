// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { Adjustments, Visible } from '@geti/ui/icons';

import { IconWrapper } from '../icon-wrapper.component';
import { ZoomFitScreen } from '../zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../zoom/zoom-selector.component';

export const Settings = () => {
    return (
        <>
            <IconWrapper>
                <Visible />
            </IconWrapper>

            <IconWrapper>
                <Adjustments />
            </IconWrapper>

            <Divider size='S' />

            <ZoomSelector />

            <Divider size='S' />

            <ZoomFitScreen />
        </>
    );
};
