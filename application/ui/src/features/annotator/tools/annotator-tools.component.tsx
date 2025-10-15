// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { BoundingBox, SegmentAnythingIcon, Selector } from '@geti/ui/icons';
import { useProjectTask } from 'hooks/use-project-task.hook';

import { ToolConfig } from '../../dataset/media-preview/primary-toolbar/tools/interface';
import { Tools } from '../../dataset/media-preview/primary-toolbar/tools/tools.component';
import { useAnnotator } from '../annotator-provider.component';

const TASK_TOOL_CONFIG: Record<string, ToolConfig[]> = {
    classification: [],
    detection: [
        { type: 'selection', icon: Selector },
        { type: 'bounding-box', icon: BoundingBox },
    ],
    segmentation: [
        { type: 'selection', icon: Selector },
        { type: 'sam', icon: SegmentAnythingIcon },
        // TODO: Add 'polygon' tool later
    ],
};

export const AnnotatorTools = () => {
    const projectTask = useProjectTask();
    const { activeTool, setActiveTool } = useAnnotator();
    const availableTools = TASK_TOOL_CONFIG[projectTask] || [];

    return (
        <>
            <Tools tools={availableTools} activeTool={activeTool} setActiveTool={setActiveTool} />
            {availableTools.length > 0 && <Divider size='S' />}
        </>
    );
};
