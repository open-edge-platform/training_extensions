// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Divider, Grid, Heading, View } from '@geti/ui';
import { ZoomProvider } from 'src/components/zoom/zoom';

import { AnnotatorCanvas } from '../../annotator/annotator-canvas';
import { SelectAnnotationProvider } from '../../annotator/select-annotation-provider.component';
import { ToolSelectionBar } from '../../annotator/tools/tool-selection-bar.component';
import { DatasetItem } from '../../annotator/types';
import { AnnotatorButtons } from './annotator-buttons.component';

type MediaPreviewProps = {
    mediaItem: DatasetItem;
    close: () => void;
};
export const MediaPreview = ({ mediaItem, close }: MediaPreviewProps) => {
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
                <View height='100%'>
                    <Grid
                        areas={['toolbar canvas aside', 'toolbar canvas aside', 'toolbar footer aside']}
                        width={'100%'}
                        height='100%'
                        columns={'100px calc(100% - 200px) 100px'}
                        rows={'auto 1fr auto'}
                    >
                        <Suspense fallback={<div>Loading...</div>}>
                            <ZoomProvider>
                                <View gridArea={'toolbar'}>
                                    <ToolSelectionBar />
                                </View>

                                <View gridArea={'canvas'}>
                                    <SelectAnnotationProvider>
                                        <AnnotatorCanvas mediaItem={mediaItem} isFocussed={isFocussed} />
                                    </SelectAnnotationProvider>
                                </View>

                                <View gridArea={'aside'}>
                                    <div>Aside</div>
                                </View>

                                <View gridArea={'footer'} padding={'size-100'} UNSAFE_style={{ textAlign: 'right' }}>
                                    <AnnotatorButtons onFocus={setIsFocussed} isFocussed={isFocussed} onClose={close} />
                                </View>
                            </ZoomProvider>
                        </Suspense>
                    </Grid>
                </View>
            </Content>
        </Dialog>
    );
};
