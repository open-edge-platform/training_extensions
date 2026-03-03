// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProject } from 'hooks/api/project.hook';

import { useTool } from '../../../shared/annotator/tool-provider.component';
import { isPrefetchEnabledForTask } from '../../project/task-type-guards';
import { BoundingBoxTool } from './bounding-box-tool/bounding-box-tool.component';
import { MagneticLasso } from './magnetic-lasso/magnetic-lasso.component';
import { PolygonTool } from './polygon-tool/polygon-tool.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';
import { usePreloadSAMWorkers } from './segment-anything-tool/use-segment-anything.hook';

export const ToolManager = () => {
    const { activeTool } = useTool();
    const { data: project } = useProject();

    const isPreloadEnabled = project !== undefined && isPrefetchEnabledForTask(project.task.task_type);

    // Preload SAM workers when the tool manager is mounted, so that the tool is ready
    // to use as soon as the user selects it. Not a huge performance gain but
    // it helps a bit.
    usePreloadSAMWorkers(isPreloadEnabled);

    if (activeTool === 'bounding-box') {
        return <BoundingBoxTool />;
    }

    if (activeTool === 'sam') {
        return <SegmentAnythingTool />;
    }

    if (activeTool === 'polygon') {
        return <PolygonTool />;
    }

    if (activeTool === 'magnetic-lasso') {
        return <MagneticLasso />;
    }

    return null;
};
