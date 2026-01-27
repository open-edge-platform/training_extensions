// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';
import { partition } from 'lodash-es';

import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { AnnotatorMode } from '../../../dataset/media-preview/secondary-toolbar/annotator-modes/mode';
import { Tools } from '../tools.component';
import { useAvailableTools } from './use-available-tools';

type AnnotatorToolsProps = {
    mode: AnnotatorMode;
};

export const AnnotatorTools = ({ mode }: AnnotatorToolsProps) => {
    const { activeTool, setActiveTool } = useAnnotator();

    const availableTools = useAvailableTools();
    const [selectionTool, otherTools] = partition(availableTools, (tool) => tool.type === 'selection');

    return (
        <>
            {selectionTool.length > 0 && (
                <>
                    <Tools tools={selectionTool} activeTool={activeTool} setActiveTool={setActiveTool} />
                    {otherTools.length > 0 && <Divider size='S' />}
                </>
            )}
            {mode === 'annotation' && otherTools.length > 0 && (
                <>
                    <Tools tools={otherTools} activeTool={activeTool} setActiveTool={setActiveTool} />
                    <Divider size='S' />
                </>
            )}
        </>
    );
};
