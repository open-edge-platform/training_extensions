// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@geti/ui';

import { Sidebar } from '../../features/inference/aside/sidebar-tabs.component';
import { Header } from '../../features/inference/header/inference-header.component';
import { StreamContainer } from '../../features/inference/stream/stream-container';

export const Inference = () => {
    return (
        <Grid
            areas={['toolbar aside', 'canvas aside']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) minmax(0, 1fr)',
                gridTemplateColumns: 'minmax(0, 1fr) auto',
                height: '100%',
                overflow: 'hidden',
                gap: '1px',
            }}
        >
            <Header />
            <StreamContainer />
            <Sidebar />
        </Grid>
    );
};
