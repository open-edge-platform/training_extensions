import { Button, ButtonGroup, Divider, Flex, Heading, Text, View } from '@geti/ui';

import { paths } from '../../router';
import Background from './../../assets/background.png';

export function Index() {
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
                <View>
                    <Flex direction='column' gap='size-400'>
                        <Flex justifyContent={'space-between'}>
                            <View>
                                <Heading level={2}>Input</Heading>
                                <Text>TODO</Text>
                            </View>
                            <View>
                                <Heading level={2}>Model</Heading>
                                <Text>TODO</Text>
                            </View>
                            <View>
                                <Heading level={2}>Output</Heading>
                                <Text>TODO</Text>
                            </View>
                        </Flex>
                        <Divider size='S' />
                        <ButtonGroup>
                            <Button href={paths.pipeline.input({})} variant='secondary' marginStart='auto'>
                                Edit
                            </Button>
                        </ButtonGroup>
                    </Flex>
                </View>
            </View>
        </View>
    );
}
