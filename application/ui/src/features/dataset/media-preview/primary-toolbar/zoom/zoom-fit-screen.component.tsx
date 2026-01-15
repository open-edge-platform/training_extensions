// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton } from '@geti/ui';
import { FitScreen } from '@geti/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { IconWrapper } from '../../../../../components/icon-wrapper/icon-wrapper.component';
import { useSetZoom } from '../../../../../components/zoom/zoom.provider';
import { HOTKEYS } from '../hotkeys/hotkeys-definition';

export const ZoomFitScreen = () => {
    const { fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    return (
        <ActionButton isQuiet onPress={fitToScreen} aria-label='Fit to screen'>
            <IconWrapper>
                <FitScreen />
            </IconWrapper>
        </ActionButton>
    );
};
