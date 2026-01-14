// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Invisible, Visible } from '@geti/ui/icons';

import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';

export const ToggleAnnotationsVisibility = () => {
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    return (
        <TooltipTrigger>
            <ActionButton aria-label={`${isVisible ? 'Hide' : 'Show'} annotations`} isQuiet onPress={toggleVisibility}>
                {isVisible ? <Visible /> : <Invisible />}
            </ActionButton>
            <Tooltip>{isVisible ? 'Hide annotations' : 'Show annotations'}</Tooltip>
        </TooltipTrigger>
    );
};
