import { View } from '@geti/ui';

import Background from './../../assets/background.png';
import { BuildPipeline } from './build-pipeline';

export function BuildPipelinePage({ submitPipeline }: { submitPipeline: () => void }) {
    return (
        <View
            backgroundColor={'gray-100'}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
                backgroundBlendMode: 'luminosity',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}
            gridArea={'content'}
            height='100%'
            width='100%'
        >
            <View maxWidth={'1024px'} marginX='auto' paddingY='size-900'>
                <BuildPipeline submit={submitPipeline} />
            </View>
        </View>
    );
}
