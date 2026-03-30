// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti-ui/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools/annotator-tools.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo/undo-redo.component';

export const PrimaryToolbar = () => {
    return (
        <Toolbar.Container data-testid={'primary-toolbar-id'}>
            <Flex direction={'column'} gap={'size-50'} alignItems={'center'}>
                <Toolbar.Section>
                    <Flex direction={'column'} gap={'size-50'}>
                        <AnnotatorTools />

                        <UndoRedo />
                    </Flex>
                </Toolbar.Section>
                <Toolbar.Section>
                    <Flex justifyContent={'center'}>
                        <ToggleAnnotationsVisibility />
                    </Flex>
                </Toolbar.Section>
            </Flex>
        </Toolbar.Container>
    );
};
