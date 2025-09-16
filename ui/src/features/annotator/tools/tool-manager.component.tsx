// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotator } from '../annotator-provider.component';
import { DrawingBox } from './drawing-box-tool/drawing-box.component';

export const ToolManager = () => {
    const { activeTool, addAnnotation } = useAnnotator();

    if (activeTool === 'bounding-box') {
        return <DrawingBox onComplete={addAnnotation} />;
    }

    return null;
};
