import { View } from '@geti/ui';

import { Gallery } from './gallery/gallery.component';
import { Toolbar } from './toolbar/toolbar.component';

export const DataCollection = () => {
    return (
        <View padding={'size-350'}>
            <Toolbar />

            <Gallery />
        </View>
    );
};
