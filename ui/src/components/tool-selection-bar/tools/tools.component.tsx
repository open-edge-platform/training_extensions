// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Divider } from '@geti/ui';
import { Polygon, SegmentAnythingIcon, Selector } from '@geti/ui/icons';

import { IconWrapper } from '../icon-wrapper.component';

export const Tools = () => {
    type ToolType = 'selection' | 'bounding-box' | 'polygon' | 'sam';
    const [activeTool, setActiveTool] = useState<ToolType>('selection');

    const handleSelectTool = (tool: ToolType) => {
        setActiveTool(tool);
    };

    return (
        <>
            <IconWrapper onPress={() => handleSelectTool('selection')} isSelected={activeTool === 'selection'}>
                <Selector data-tool='selection' />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper onPress={() => handleSelectTool('polygon')} isSelected={activeTool === 'polygon'}>
                <Polygon data-tool='polygon' />
            </IconWrapper>

            <IconWrapper onPress={() => handleSelectTool('sam')} isSelected={activeTool === 'sam'}>
                <SegmentAnythingIcon data-tool='sam' />
            </IconWrapper>
        </>
    );
};
