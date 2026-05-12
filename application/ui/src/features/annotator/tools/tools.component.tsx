// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import type { ToolConfig, ToolType } from './interface';
import { Tool } from './tool/tool.component';

interface ToolsProps {
    tools: ToolConfig[];
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
    isDisabled?: boolean;
}

export const Tools = ({ tools, activeTool, setActiveTool, isDisabled }: ToolsProps) => {
    if (tools.length === 0) {
        return null;
    }

    return (
        <>
            {tools.map((tool) => (
                <Fragment key={tool.type}>
                    <Tool tool={tool} activeTool={activeTool} setActiveTool={setActiveTool} isDisabled={isDisabled} />
                </Fragment>
            ))}
        </>
    );
};
