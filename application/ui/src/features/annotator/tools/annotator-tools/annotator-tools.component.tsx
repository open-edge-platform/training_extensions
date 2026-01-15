// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider } from '@geti/ui';

import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { Tools } from '../tools.component';
import { useAvailableTools } from './use-available-tools';

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    const availableTools = useAvailableTools();

    return (
        <>
            <Tools tools={availableTools} activeTool={activeTool} setActiveTool={setActiveTool} />
            {availableTools.length > 0 && <Divider size='S' />}
        </>
    );
};
