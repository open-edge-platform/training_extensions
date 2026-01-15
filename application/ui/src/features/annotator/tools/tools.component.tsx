// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useHotkeys } from 'react-hotkeys-hook';
import { Fragment } from 'react/jsx-runtime';

import { IconWrapper } from '../../../components/icon-wrapper/icon-wrapper.component';
import type { ToolConfig, ToolType } from './interface';

interface ToolProps {
    tool: ToolConfig;
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
}

const Tool = ({ tool, activeTool, setActiveTool }: ToolProps) => {
    useHotkeys(tool.hotkey, () => setActiveTool(tool.type), [setActiveTool]);

    return (
        <IconWrapper onPress={() => setActiveTool(tool.type)} isSelected={activeTool === tool.type}>
            <tool.icon data-tool={tool.type} />
        </IconWrapper>
    );
};

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
                    <Tool tool={tool} activeTool={activeTool} setActiveTool={setActiveTool} />
                </Fragment>
            ))}
        </>
    );
};
