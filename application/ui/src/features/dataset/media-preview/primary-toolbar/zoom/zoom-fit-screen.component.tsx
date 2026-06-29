// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { FitScreen } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { IconWrapper } from '../../../../../components/icon-wrapper/icon-wrapper.component';
import { useSetZoom } from '../../../../../components/zoom/zoom.provider';
import { HOTKEYS } from '../../../../../shared/hotkeys-definition';

export const ZoomFitScreen = () => {
    const { fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    return (
        <TooltipTrigger>
            <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit to screen'>
                <IconWrapper>
                    <FitScreen />
                </IconWrapper>
            </ActionButton>
            <Tooltip>Fit to screen</Tooltip>
        </TooltipTrigger>
    );
};
