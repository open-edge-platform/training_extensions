// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { BoundingBox, Selector } from '@geti/ui/icons';

import { useAnnotator } from '../../../features/annotator/annotator-provider.component';
import { IconWrapper } from '../icon-wrapper.component';

export const Tools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    return (
        <>
            <IconWrapper onPress={() => setActiveTool('selection')} isSelected={activeTool === 'selection'}>
                <Selector data-tool='selection' />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper onPress={() => setActiveTool('bounding-box')} isSelected={activeTool === 'bounding-box'}>
                <BoundingBox />
            </IconWrapper>
        </>
    );
};
