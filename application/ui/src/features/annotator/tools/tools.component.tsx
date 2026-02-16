// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { useHotkeys } from 'react-hotkeys-hook';
import { Fragment } from 'react/jsx-runtime';

import { IconWrapper } from '../../../components/icon-wrapper/icon-wrapper.component';
import type { ToolConfig, ToolType } from './interface';

interface ToolProps {
    tool: ToolConfig;
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
    isDisabled?: boolean;
}

const Tool = ({ tool, activeTool, setActiveTool, isDisabled }: ToolProps) => {
    useHotkeys(tool.hotkey, () => setActiveTool(tool.type), [setActiveTool]);

    return (
        <TooltipTrigger placement={'right'}>
            <ActionButton
                isQuiet
                width={'size-400'}
                onPress={() => setActiveTool(tool.type)}
                aria-label={`${tool.type} tool`}
                isDisabled={isDisabled}
            >
                <IconWrapper isSelected={activeTool === tool.type} isDisabled={isDisabled}>
                    <tool.icon data-tool={tool.type} />
                </IconWrapper>
            </ActionButton>
            <Tooltip>
                {tool.label} ({tool.hotkey})
            </Tooltip>
        </TooltipTrigger>
    );
};

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
