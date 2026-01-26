// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';

import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { Tools } from '../tools.component';
import { useAvailableTools } from './use-available-tools';

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    const availableTools = useAvailableTools();
    const selectionTool = availableTools.find((tool) => tool.type === 'selection');
    const otherTools = availableTools.filter((tool) => tool.type !== 'selection');

    return (
        <>
            {selectionTool && (
                <>
                    <Tools tools={[selectionTool]} activeTool={activeTool} setActiveTool={setActiveTool} />
                    {otherTools.length > 0 && <Divider size='S' />}
                </>
            )}
            {otherTools.length > 0 && (
                <Tools tools={otherTools} activeTool={activeTool} setActiveTool={setActiveTool} />
            )}
        </>
    );
};
