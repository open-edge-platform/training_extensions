// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ToolType } from '../../../components/tool-selection-bar/tools/interface';

export const useActiveTool = () => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');

    const handleSelectTool = (tool: ToolType) => {
        setActiveTool(tool);
    };

    return { activeTool, selectTool: handleSelectTool };
};
