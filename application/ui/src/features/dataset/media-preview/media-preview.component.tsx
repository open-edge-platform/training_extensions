// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { Content, Dialog, Grid, View } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import type { Media } from '../../../constants/shared-types';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { AnnotatorProviders } from './annotator-providers.component';
import { AnnotatorContainer } from './annotator.component';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { SIDEBAR_WIDTH } from './constants';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { getInitialAnnotations, getInitialPredictions } from './utils';

type MediaPreviewProps = {
    mediaItem: Media;
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

type MediaPreviewContentProps = {
    mediaItem: Media;
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const MediaPreviewContent = ({ mediaItem, items, onSelectedMediaItem, onClose }: MediaPreviewContentProps) => {
    const [mode, setMode] = useState<AnnotatorMode>('annotation');

    const { data: annotationsData } = useAnnotationsQuery(mediaItem);

    const isUserReviewed = annotationsData?.user_reviewed ?? false;

    const initialAnnotations = useMemo(() => {
        return getInitialAnnotations(isUserReviewed, annotationsData?.annotations ?? []);
    }, [isUserReviewed, annotationsData?.annotations]);

    const initialPredictions = useMemo(() => {
        return getInitialPredictions(isUserReviewed, annotationsData?.annotations ?? []);
    }, [isUserReviewed, annotationsData?.annotations]);

    return (
        <ToolProvider mode={mode}>
            <AnnotatorProviders
                mediaItem={mediaItem}
                initialAnnotationsDTO={initialAnnotations}
                initialPredictionsDTO={initialPredictions}
                isUserReviewed={isUserReviewed}
                mode={mode}
            >
                <AnnotatorContainer
                    mode={mode}
                    items={items}
                    onClose={onClose}
                    changeAnnotatorMode={setMode}
                    onSelectedMediaItem={onSelectedMediaItem}
                />
            </AnnotatorProviders>
        </ToolProvider>
    );
};

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetMediaItems();

    return (
        <Dialog
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
                '--spectrum-dialog-padding-x': 'var(--spectrum-global-dimension-size-250)',
                '--spectrum-dialog-padding-y': 'var(--spectrum-global-dimension-size-250)',
            }}
        >
            <Content>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto auto'
                    columns={['size-700', 'minmax(0, 1fr)', SIDEBAR_WIDTH]}
                    areas={[
                        'header header aside',
                        'toolbar canvas aside',
                        'toolbar video-toolbar aside',
                        'toolbar bottom aside',
                    ]}
                >
                    <MediaPreviewContent
                        mediaItem={mediaItem}
                        items={items}
                        onClose={close}
                        onSelectedMediaItem={onSelectedMediaItem}
                    />

                    <View gridArea={'aside'}>
                        <SidebarItems
                            items={items}
                            mediaItem={mediaItem}
                            hasNextPage={hasNextPage}
                            isFetchingNextPage={isFetchingNextPage}
                            fetchNextPage={fetchNextPage}
                            onSelectedMediaItem={onSelectedMediaItem}
                        />
                    </View>
                </Grid>
            </Content>
        </Dialog>
    );
};
