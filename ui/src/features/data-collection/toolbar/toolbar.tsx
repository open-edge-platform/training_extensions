import { Button, Flex, Heading, Text, View } from '@geti/ui';

export const Toolbar = () => {
    return (
        <Flex justifyContent={'space-between'} alignItems={'center'} gridArea={'toolbar'}>
            <View>
                <Heading level={4} marginY='size-100'>
                    Confidence Threshold
                </Heading>
                <Text>Below 0.1</Text>
            </View>
            <View>
                <Heading level={4} marginY='size-100'>
                    Drift score
                </Heading>
                <Text>Lorem ipsum</Text>
            </View>

            <View>
                <Heading level={4} marginY='size-100'>
                    Test Rate
                </Heading>
                <Text>Check one frame every 10 seconds</Text>
            </View>

            <View>
                <Heading level={4} marginY='size-100'>
                    Max Frames to Collect
                </Heading>
                <Text>5,000</Text>
            </View>
            <Flex height='100%' alignItems={'center'}>
                <Button variant='secondary'>Edit collection criteria</Button>
            </Flex>
        </Flex>
    );
};
