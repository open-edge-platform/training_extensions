// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton } from '@geti/ui';
import { FitScreen } from '@geti/ui/icons';

import { useSetZoom } from '../../../../../components/zoom/zoom.provider';
import { IconWrapper } from '../icon-wrapper.component';

export const ZoomFitScreen = () => {
    const { fitToScreen } = useSetZoom();

    return (
        <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit to screen'>
            <IconWrapper>
                <FitScreen />
            </IconWrapper>
        </ActionButton>
    );
};
