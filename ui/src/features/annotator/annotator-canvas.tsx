// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import thumbnailUrl from '../../assets/mocked-project-thumbnail.png';
import { ZoomProvider } from '../../components/zoom/zoom';
import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { response } from '../dataset/mock-response';
import { Annotations } from './annotations';

type Item = (typeof response.items)[number];

export const AnnotatorCanvas = ({ mediaItem, isFocussed }: { mediaItem: Item; isFocussed: boolean }) => {
    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomProvider>
            <ZoomTransform target={size}>
                <Grid areas={['innercanvas']} width={'100%'} height='100%'>
                    <View gridArea={'innercanvas'}>
                        <img src={thumbnailUrl} alt='Collected data' />
                    </View>

                    {!isEmpty(mediaItem.annotations) && (
                        <View gridArea={'innercanvas'}>
                            <Annotations
                                annotations={mediaItem.annotations}
                                width={size.width}
                                height={size.height}
                                isFocussed={isFocussed}
                            />
                        </View>
                    )}
                </Grid>
            </ZoomTransform>
        </ZoomProvider>
    );
};
