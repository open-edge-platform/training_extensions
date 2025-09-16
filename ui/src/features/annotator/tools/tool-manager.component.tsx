// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotator } from '../annotator-provider.component';
import { BoundingBoxTool } from './bounding-box-tool/bounding-box-tool.component';

export const ToolManager = () => {
    const { activeTool } = useAnnotator();

    if (activeTool === 'bounding-box') {
        return <BoundingBoxTool />;
    }

    return null;
};
