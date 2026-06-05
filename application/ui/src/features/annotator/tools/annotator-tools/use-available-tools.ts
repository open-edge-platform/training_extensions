// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BoundingBox, Polygon, SegmentAnythingIcon, Selector } from '@geti/ui/icons';

import { ReactComponent as MagneticLasso } from '../../../../assets/icons/magnetic-lasso.svg';
import BoundingBoxImg from '../../../../assets/tools/bounding-box.webp';
import MagneticLassoImg from '../../../../assets/tools/magnetic-lasso.webp';
import PolygonImg from '../../../../assets/tools/polygon.webp';
import SAMDetectionImg from '../../../../assets/tools/sam-detection.webp';
import SAMSegmentationImg from '../../../../assets/tools/sam-segmentation.webp';
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
    tooltip: {
        img: BoundingBoxImg,
        description: 'Draw a rectangle or square surrounding an object in an image.',
    },
};

const AUTO_SEGMENTATION_DETECTION_CONFIG: ToolConfig = {
    type: 'sam',
    icon: SegmentAnythingIcon,
    hotkey: HOTKEYS.autoSegmentation,
    label: 'Auto segmentation',
    tooltip: {
        img: SAMDetectionImg,
        description:
            'Move your cursor over an object to preview a suggested rectangle, then click to create the annotation.',
    },
};

const AUTO_SEGMENTATION_CONFIG: ToolConfig = {
    type: 'sam',
    icon: SegmentAnythingIcon,
    hotkey: HOTKEYS.autoSegmentation,
    label: 'Auto segmentation',
    tooltip: {
        img: SAMSegmentationImg,
        description:
            'Move your cursor over an object to preview a suggested polygon, then click to create the annotation.',
    },
};

const POLYGON_TOOL_CONFIG: ToolConfig = {
    type: 'polygon',
    icon: Polygon,
    hotkey: HOTKEYS.polygonTool,
    label: 'Polygon',
    tooltip: {
        img: PolygonImg,
        description:
            'Click to place points one by one, or hold and drag to draw freehand. Ideal for irregular shapes ' +
            'requiring pixel-precision.',
    },
};

const MAGNETIC_LASSO_TOOL_CONFIG: ToolConfig = {
    type: 'magnetic-lasso',
    icon: MagneticLasso,
    hotkey: HOTKEYS.magneticLassoTool,
    label: 'Magnetic Lasso',
    tooltip: {
        img: MagneticLassoImg,
        description: "Hover along an object's edge and click periodically to snap the outline to its borders.",
    },
};

// TODO: Disable for 3.0, enable for 3.1 after improvements (needs a sidebar to tweak threshold)
// const SSIM_TOOL_CONFIG: ToolConfig = {
//     type: 'ssim',
//     icon: DetectionTool,
//     hotkey: HOTKEYS.ssimTool,
//     label: 'Detection assistant',
// };

const TASK_TOOL_CONFIG: Record<string, ToolConfig[]> = {
    classification: [],
    detection: [SELECTION_TOOL_CONFIG, BOUNDING_BOX_TOOL_CONFIG, AUTO_SEGMENTATION_DETECTION_CONFIG],
    instance_segmentation: [
        SELECTION_TOOL_CONFIG,
        POLYGON_TOOL_CONFIG,
        MAGNETIC_LASSO_TOOL_CONFIG,
        AUTO_SEGMENTATION_CONFIG,
        // TODO: Disable for 3.0, enable for 3.1 after improvements (needs a sidebar to tweak threshold)
        // SSIM_TOOL_CONFIG,
    ],
};

export const useAvailableTools = (): ToolConfig[] => {
    const taskType = useProjectTask();

    return TASK_TOOL_CONFIG[taskType];
};
