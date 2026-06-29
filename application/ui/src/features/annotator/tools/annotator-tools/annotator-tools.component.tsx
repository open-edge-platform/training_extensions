// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti-ui/ui';
import { partition } from 'lodash-es';

import { useTool } from '../../../../shared/annotator/tool-provider.component';
import { isNonEmptyArray } from '../../../../shared/util';
import { useIsAnnotatorSceneBusy } from '../../hooks/use-is-annotator-scene-busy';
import { Tools } from '../tools.component';
import { useAvailableTools } from './use-available-tools';

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useTool();
    const isSceneBusy = useIsAnnotatorSceneBusy();

    const availableTools = useAvailableTools();
    const [selectionTool, otherTools] = partition(availableTools, (tool) => tool.type === 'selection');

    return (
        <>
            {isNonEmptyArray(selectionTool) && (
                <>
                    <Tools
                        tools={selectionTool}
                        activeTool={activeTool}
                        setActiveTool={setActiveTool}
                        isDisabled={isSceneBusy}
                    />
                    {isNonEmptyArray(otherTools) && <Divider size='S' />}
                </>
            )}
            {isNonEmptyArray(otherTools) && (
                <>
                    <Tools
                        tools={otherTools}
                        activeTool={activeTool}
                        setActiveTool={setActiveTool}
                        isDisabled={isSceneBusy}
                    />
                    <Divider size='S' />
                </>
            )}
        </>
    );
};
