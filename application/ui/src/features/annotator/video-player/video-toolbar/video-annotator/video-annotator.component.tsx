// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, View } from '@geti/ui';
import { useProjectLabels } from 'hooks/use-project-labels.hook';

import type { AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';
import { Labels } from './labels/labels.component';
import { VideoTimeline } from './video-timeline/video-timeline.component';

type VideoAnnotatorProps = {
    mode: AnnotatorMode;
};

export const VideoAnnotator = ({ mode }: VideoAnnotatorProps) => {
    const labels = useProjectLabels();

    return (
        <Grid
            columns={['size-2000', 'auto']}
            rows={['size-500', 'auto']}
            areas={['. timeline', 'labels timeline']}
            maxHeight={'size-2000'}
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            <View gridArea={'labels'}>
                <Labels labels={labels} />
            </View>
            <View gridArea={'timeline'} UNSAFE_style={{ overflowX: 'auto' }}>
                <VideoTimeline labels={labels} mode={mode} />
            </View>
        </Grid>
    );
};
