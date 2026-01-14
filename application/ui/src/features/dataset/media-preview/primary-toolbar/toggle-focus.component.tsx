// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LabelGroup } from '@geti/ui/icons';

import { IconWrapper } from '../../../../components/icon-wrapper/icon-wrapper.component';
import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';

export const ToggleFocus = () => {
    const { toggleFocus, isFocussed } = useAnnotationVisibility();

    return (
        <IconWrapper onPress={toggleFocus} isSelected={isFocussed}>
            <LabelGroup />
        </IconWrapper>
    );
};
