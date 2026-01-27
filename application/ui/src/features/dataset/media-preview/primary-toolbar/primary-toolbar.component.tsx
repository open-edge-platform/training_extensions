// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools/annotator-tools.component';
import { AnnotatorMode } from '../secondary-toolbar/annotator-modes/mode';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo/undo-redo.component';

type PrimaryToolbarProps = {
    mode: AnnotatorMode;
};

export const PrimaryToolbar = ({ mode }: PrimaryToolbarProps) => {
    return (
        <Toolbar.Container>
            <Flex direction={'column'} gap={'size-50'}>
                <Toolbar.Section>
                    <Flex direction={'column'} gap={'size-50'}>
                        <AnnotatorTools mode={mode} />

                        <UndoRedo />
                    </Flex>
                </Toolbar.Section>
                <Toolbar.Section>
                    <ToggleAnnotationsVisibility />
                </Toolbar.Section>
            </Flex>
        </Toolbar.Container>
    );
};
