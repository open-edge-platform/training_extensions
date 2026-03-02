// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react';

import { Content, Dialog, Grid, View } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { Remote } from 'comlink';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Media } from '../../../constants/shared-types';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { getLoadImageQueryKey, loadImageQueryFn } from '../../annotator/hooks/use-load-image-query.hook';
import {
    SelectedMediaItemProvider,
    useSelectedMediaItem,
} from '../../annotator/selected-media-item-provider.component';
import {
    getSegmentAnythingEncodingQueryKey,
    getSegmentAnythingWorkerQueryKey,
    segmentAnythingEncodingQueryFn,
    segmentAnythingWorkerQueryFn,
} from '../../annotator/tools/segment-anything-tool/use-segment-anything.hook';
import type { SegmentAnythingWorkerModel } from '../../annotator/webworkers/segment-anything.worker.interface';
import { AnnotatorProviders } from './annotator-providers.component';
import { AnnotatorContainer } from './annotator.component';
import { annotationsQueryFn, getAnnotationsQueryKey, useAnnotationsQuery } from './api/use-annotations-query';
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
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

// When the user navigates to next media, the most expensive data, like the SAM encoding,
// along with image data and annotations, will be already in React Query cache, so the UI will feel smoother
// whenever the user switches image. Unless he/she changes to a random or item. We could also consider
// those cases but I feel like it's overkill. Let's see how this improvement performs and then we can iterate on it.
//
// ensureQueryData will get the data from cache if it's there, or call the queryFn and cache the result if it's not.
// prefetchQuery will fetch the data and cache it
const prefetchNextMediaItemData = ({
    queryClient,
    projectId,
    items,
    selectedItem,
}: {
    queryClient: ReturnType<typeof useQueryClient>;
    projectId: string;
    items: Media[];
    selectedItem: Media;
}) => {
    const prefetch = async () => {
        const selectedIndex = items.findIndex((item) => item.id === selectedItem.id);
        const nextItem = selectedIndex >= 0 ? items[selectedIndex + 1] : undefined;

        if (nextItem === undefined) {
            return;
        }

        const nextImage = await queryClient.ensureQueryData({
            queryKey: getLoadImageQueryKey(projectId, nextItem),
            queryFn: () => loadImageQueryFn(projectId, nextItem),
            staleTime: Infinity,
            retry: 0,
        });

        queryClient.prefetchQuery({
            queryKey: getAnnotationsQueryKey(projectId, nextItem),
            queryFn: () => annotationsQueryFn(projectId, nextItem),
        });

        const encoderModel = await queryClient.ensureQueryData<Remote<SegmentAnythingWorkerModel>>({
            queryKey: getSegmentAnythingWorkerQueryKey('SEGMENT_ANYTHING_ENCODER'),
            queryFn: segmentAnythingWorkerQueryFn('SEGMENT_ANYTHING_ENCODER'),
            staleTime: Infinity,
        });

        queryClient.prefetchQuery({
            queryKey: getSegmentAnythingEncodingQueryKey(nextItem),
            queryFn: () => segmentAnythingEncodingQueryFn(encoderModel, nextImage),
            staleTime: Infinity,
            gcTime: 3600 * 15,
        });
    };

    prefetch();
};

const MediaPreviewContent = ({ items, onSelectedMediaItem, onClose }: MediaPreviewContentProps) => {
    const [mode, setMode] = useState<AnnotatorMode>('annotation');
    const { mediaItem } = useSelectedMediaItem();
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();

    const { data: annotationsData } = useAnnotationsQuery(mediaItem);

    const isUserReviewed = annotationsData?.user_reviewed ?? false;

    const initialAnnotations = useMemo(() => {
        return getInitialAnnotations(isUserReviewed, annotationsData?.annotations ?? []);
    }, [isUserReviewed, annotationsData?.annotations]);

    const initialPredictions = useMemo(() => {
        return getInitialPredictions(isUserReviewed, annotationsData?.annotations ?? []);
    }, [isUserReviewed, annotationsData?.annotations]);

    useEffect(() => {
        prefetchNextMediaItemData({
            queryClient,
            projectId,
            items,
            selectedItem: mediaItem,
        });
    }, [items, mediaItem, projectId, queryClient]);

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
                    <SelectedMediaItemProvider mediaItem={mediaItem}>
                        <MediaPreviewContent items={items} onClose={close} onSelectedMediaItem={onSelectedMediaItem} />
                    </SelectedMediaItemProvider>

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
