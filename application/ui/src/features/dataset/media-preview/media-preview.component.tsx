// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Divider, Grid, Heading, View } from '@geti/ui';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas';
import { SelectAnnotationProvider } from '../../annotator/select-annotation-provider.component';
import { DatasetItem } from '../../annotator/types';
import { AnnotatorButtons } from './annotator-buttons.component';
import { ToolSelectionBar } from './primary-toolbar/primary-toolbar.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';

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
                <Grid
                    areas={['toolbar header aside', 'toolbar canvas aside', 'toolbar footer aside']}
                    width={'100%'}
                    height={'100%'}
                    columns={'100px 1fr 100px'}
                    rows={'auto 1fr auto'}
                >
                    <Suspense fallback={<div>Loading...</div>}>
                        <ZoomProvider>
                            <SelectAnnotationProvider>
                                <View gridArea={'toolbar'}>
                                    <ToolSelectionBar />
                                </View>

                                <View gridArea={'header'}>
                                    <SecondaryToolbar />
                                </View>
                                <View gridArea={'canvas'} overflow={'hidden'}>
                                    <AnnotatorCanvas mediaItem={mediaItem} isFocussed={isFocussed} />
                                </View>
                            </SelectAnnotationProvider>

                            <View gridArea={'aside'}>
                                <div>Aside</div>
                            </View>

                            <View gridArea={'footer'} padding={'size-100'} UNSAFE_style={{ textAlign: 'right' }}>
                                <AnnotatorButtons onFocus={setIsFocussed} isFocussed={isFocussed} onClose={close} />
                            </View>
                        </ZoomProvider>
                    </Suspense>
                </Grid>
            </Content>
        </Dialog>
    );
};
