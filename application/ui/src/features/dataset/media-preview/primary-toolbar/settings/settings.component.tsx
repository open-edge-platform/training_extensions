// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { Adjustments, Invisible, LabelGroup, Visible } from '@geti/ui/icons';
import { useAnnotationVisibility } from 'src/features/annotator/annotation-visibility-provider.component';

import { IconWrapper } from '../icon-wrapper.component';
import { ZoomFitScreen } from '../zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../zoom/zoom-selector.component';

export const Settings = () => {
    const { isVisible, toggleVisibility, isFocussed, toggleFocus } = useAnnotationVisibility();

    return (
        <>
            <IconWrapper onPress={toggleVisibility}>{isVisible ? <Visible /> : <Invisible />}</IconWrapper>

            <IconWrapper>
                <Adjustments />
            </IconWrapper>

            <Divider size='S' />

            <ZoomSelector />

            <Divider size='S' />

            <IconWrapper onPress={toggleFocus} isSelected={isFocussed}>
                <LabelGroup />
            </IconWrapper>

            <ZoomFitScreen />
        </>
    );
};
