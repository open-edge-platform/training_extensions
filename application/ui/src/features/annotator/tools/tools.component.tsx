// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react/jsx-runtime';

import { IconWrapper } from '../../../components/icon-wrapper/icon-wrapper.component';
import { ToolConfig, ToolType } from './interface';

interface ToolsProps {
    tools: ToolConfig[];
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
}
export const Tools = ({ tools, activeTool, setActiveTool }: ToolsProps) => {
    if (tools.length === 0) {
        return null;
    }

    return (
        <>
            {tools.map((tool) => (
                <Fragment key={tool.type}>
                    <IconWrapper onPress={() => setActiveTool(tool.type)} isSelected={activeTool === tool.type}>
                        <tool.icon data-tool={tool.type} />
                    </IconWrapper>
                </Fragment>
            ))}
        </>
    );
};
