// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Divider, Flex, Grid, Heading, Loading, View } from '@geti/ui';
import { AnnotationActionsProvider } from 'src/features/annotator/annotation-actions-provider.component';
import { AnnotatorProvider } from 'src/features/annotator/annotator-provider.component';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas';
import { SelectAnnotationProvider } from '../../annotator/select-annotation-provider.component';
import { DatasetItem } from '../../annotator/types';
import { AnnotatorButtons } from './annotator-buttons.component';
import { ToolSelectionBar } from './primary-toolbar/primary-toolbar.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items.component';

type MediaPreviewProps = {
    mediaItem: DatasetItem;
    close: () => void;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

const CanvasAreaLoading = () => (
    <Flex gridArea={'canvas'} alignContent={'center'} justifyContent={'center'}>
        <Loading size='L' mode='inline' />
    </Flex>
);

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const [isFocussed, setIsFocussed] = useState(false);

    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>

            <Divider />

            <Content
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-50)',
                }}
            >
                <Grid
                    areas={['toolbar header aside', 'toolbar canvas aside', 'toolbar footer aside']}
                    width={'100%'}
                    height='100%'
                    columns={'100px calc(100% - 318px) 218px'}
                    rows={'auto 1fr auto'}
                >
                    <AnnotationActionsProvider mediaItem={mediaItem}>
                        <ZoomProvider>
                            <Suspense fallback={<CanvasAreaLoading />}>
                                <SelectAnnotationProvider>
                                    <AnnotatorProvider mediaItem={mediaItem}>
                                        <View gridArea={'toolbar'}>
                                            <ToolSelectionBar />
                                        </View>

                                        <View gridArea={'header'}>
                                            <SecondaryToolbar />
                                        </View>
                                        <View gridArea={'canvas'} overflow={'hidden'}>
                                            <AnnotatorCanvas mediaItem={mediaItem} isFocussed={isFocussed} />
                                        </View>
                                    </AnnotatorProvider>
                                </SelectAnnotationProvider>
                            </Suspense>

                            <View gridArea={'aside'}>
                                <SidebarItems mediaItem={mediaItem} onSelectedMediaItem={onSelectedMediaItem} />
                            </View>

                            <View gridArea={'footer'} padding={'size-100'} UNSAFE_style={{ textAlign: 'right' }}>
                                <AnnotatorButtons onFocus={setIsFocussed} isFocussed={isFocussed} onClose={close} />
                            </View>
                        </ZoomProvider>
                    </AnnotationActionsProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
