// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { Fragment } from 'react/jsx-runtime';

import { IconWrapper } from '../icon-wrapper.component';
import { ToolConfig, ToolType } from './interface';

interface ToolsProps {
    tools: ToolConfig[];
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
}
export const Tools = ({ tools, activeTool, setActiveTool }: ToolsProps) => {
    return (
        <>
            {tools.map((tool, index) => (
                <Fragment key={tool.type}>
                    {index > 0 && <Divider size='S' />}

                    <IconWrapper onPress={() => setActiveTool(tool.type)} isSelected={activeTool === tool.type}>
                        <tool.icon data-tool={tool.type} />
                    </IconWrapper>
                </Fragment>
            ))}
        </>
    );
};
