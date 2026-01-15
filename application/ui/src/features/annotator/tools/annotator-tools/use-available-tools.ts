// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BoundingBox, Polygon, SegmentAnythingIcon, Selector } from '@geti/ui/icons';
import { useProjectTask } from 'hooks/use-project-task.hook';

import { HOTKEYS } from '../../../dataset/media-preview/primary-toolbar/hotkeys/hotkeys-definition';
import { ToolConfig } from '../interface';

const SELECTION_TOOL_CONFIG: ToolConfig = {
    type: 'selection',
    icon: Selector,
    hotkey: HOTKEYS.selectionTool,
    label: 'Selection',
};
const BOUNDING_BOX_TOOL_CONFIG: ToolConfig = {
    type: 'bounding-box',
    icon: BoundingBox,
    hotkey: HOTKEYS.boundingBoxTool,
    label: 'Bounding box',
};
const POLYGON_TOOL_CONFIG: ToolConfig = {
    type: 'polygon',
    icon: Polygon,
    hotkey: HOTKEYS.polygonTool,
    label: 'Polygon',
};
const AUTO_SEGMENTATION_CONFIG: ToolConfig = {
    type: 'sam',
    icon: SegmentAnythingIcon,
    hotkey: HOTKEYS.autoSegmentation,
    label: 'Auto segmentation',
};

const TASK_TOOL_CONFIG: Record<string, ToolConfig[]> = {
    classification: [],
    detection: [SELECTION_TOOL_CONFIG, BOUNDING_BOX_TOOL_CONFIG],
    instance_segmentation: [SELECTION_TOOL_CONFIG, AUTO_SEGMENTATION_CONFIG, POLYGON_TOOL_CONFIG],
};

export const useAvailableTools = (): ToolConfig[] => {
    const taskType = useProjectTask();

    return TASK_TOOL_CONFIG[taskType];
};
