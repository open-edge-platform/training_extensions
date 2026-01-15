// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Invisible, Visible } from '@geti/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { HOTKEYS } from './hotkeys/hotkeys-definition';

export const ToggleAnnotationsVisibility = () => {
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    useHotkeys(HOTKEYS.toggleAnnotationsVisibility, toggleVisibility, [toggleVisibility]);

    return (
        <TooltipTrigger>
            <ActionButton aria-label={`${isVisible ? 'Hide' : 'Show'} annotations`} isQuiet onPress={toggleVisibility}>
                {isVisible ? <Visible /> : <Invisible />}
            </ActionButton>
            <Tooltip>{isVisible ? 'Hide annotations' : 'Show annotations'}</Tooltip>
        </TooltipTrigger>
    );
};
