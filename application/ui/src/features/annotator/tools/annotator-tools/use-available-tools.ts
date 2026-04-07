// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BoundingBox, DetectionTool, Polygon, SegmentAnythingIcon, Selector } from '@geti/ui/icons';

import { ReactComponent as MagneticLasso } from '../../../../assets/icons/magnetic-lasso.svg';
import { useProjectTask } from '../../../../hooks/use-project-task.hook';
import { HOTKEYS } from '../../../../shared/hotkeys-definition';
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
const AUTO_SEGMENTATION_CONFIG: ToolConfig = {
    type: 'sam',
    icon: SegmentAnythingIcon,
    hotkey: HOTKEYS.autoSegmentation,
    label: 'Auto segmentation',
};
const POLYGON_TOOL_CONFIG: ToolConfig = {
    type: 'polygon',
    icon: Polygon,
    hotkey: HOTKEYS.polygonTool,
    label: 'Polygon',
};
const MAGNETIC_LASSO_TOOL_CONFIG: ToolConfig = {
    type: 'magnetic-lasso',
    icon: MagneticLasso,
    hotkey: HOTKEYS.magneticLassoTool,
    label: 'Magnetic Lasso',
};
const SSIM_TOOL_CONFIG: ToolConfig = {
    type: 'ssim',
    icon: DetectionTool,
    hotkey: HOTKEYS.ssimTool,
    label: 'SSIM',
};

const TASK_TOOL_CONFIG: Record<string, ToolConfig[]> = {
    classification: [],
    detection: [SELECTION_TOOL_CONFIG, BOUNDING_BOX_TOOL_CONFIG, AUTO_SEGMENTATION_CONFIG],
    instance_segmentation: [
        SELECTION_TOOL_CONFIG,
        POLYGON_TOOL_CONFIG,
        MAGNETIC_LASSO_TOOL_CONFIG,
        AUTO_SEGMENTATION_CONFIG,
        SSIM_TOOL_CONFIG,
    ],
};

export const useAvailableTools = (): ToolConfig[] => {
    const taskType = useProjectTask();

    return TASK_TOOL_CONFIG[taskType];
};
