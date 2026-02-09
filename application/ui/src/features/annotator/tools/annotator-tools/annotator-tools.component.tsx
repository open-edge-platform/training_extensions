// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { partition } from 'lodash-es';

import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { isNonEmptyArray } from '../../../../shared/util';
import { Tools } from '../tools.component';
import { useAvailableTools } from './use-available-tools';

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    const availableTools = useAvailableTools();
    const [selectionTool, otherTools] = partition(availableTools, (tool) => tool.type === 'selection');

    return (
        <>
            {isNonEmptyArray(selectionTool) && (
                <>
                    <Tools tools={selectionTool} activeTool={activeTool} setActiveTool={setActiveTool} />
                    {isNonEmptyArray(otherTools) && <Divider size='S' />}
                </>
            )}
            {isNonEmptyArray(otherTools) && (
                <>
                    <Tools tools={otherTools} activeTool={activeTool} setActiveTool={setActiveTool} />
                    <Divider size='S' />
                </>
            )}
        </>
    );
};
