import { useState } from 'react';

import { Grid, View } from '@geti/ui';

import { Stream } from './../../components/stream/stream';
import { ZoomTransform } from './../../components/zoom/zoom-transform';
import { Aside } from './aside';
import { Toolbar } from './toolbar';

import classes from './live-feed.module.css';

export function LiveFeedPage() {
    const [size, setSize] = useState({ height: 608, width: 892 });
    return (
        <Grid
            areas={['toolbar aside', 'canvas aside']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                gridTemplateColumns: 'auto min-content',
            }}
            height={'100%'}
            gap='1px'
        >
            <Aside />
            <Toolbar />

            <View gridArea={'canvas'} overflow={'hidden'}>
                <ZoomTransform target={size}>
                    <div className={classes.canvasContainer}>
                        <Stream size={size} setSize={setSize} />
                    </div>
                </ZoomTransform>
            </View>
        </Grid>
    );
}
