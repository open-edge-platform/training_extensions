// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotator } from '../annotator-provider.component';

export const ToolManager = () => {
    const { activeTool } = useAnnotator();

    if (activeTool === 'bounding-box') {
        // TODO: return drawing box here
    }

    return null;
};
