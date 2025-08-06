import { View } from '@geti/ui';

import { Gallery } from '../../features/data-collection/gallery/gallery';
import { Toolbar } from '../../features/data-collection/toolbar/toolbar';

export const DataCollection = () => {
    return (
        <View padding={'size-350'}>
            <Toolbar />

            <Gallery />
        </View>
    );
};
