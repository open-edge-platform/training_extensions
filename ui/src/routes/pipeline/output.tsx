import { Button, ButtonGroup, Checkbox, Divider, Flex, Form, Grid, Heading, Radio, RadioGroup, View } from '@geti/ui';
import { GridList, GridListItem } from 'react-aria-components';

import { paths } from '../../router';

export const Output = () => {
    return (
        <Form width='100%'>
            <Flex direction='column' gap='size-400'>
                <Flex alignSelf={'center'} direction='column' maxWidth='size-6000'>
                    <p>
                        Define how and where the system should deliver its predictions. This configuration enables
                        seamless integration with your existing workflows and infrastructure.
                    </p>
                    <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                        Trigger
                    </Heading>
                    <RadioGroup label='Trigger' isHidden>
                        <Radio value='Low score'>Low score</Radio>
                    </RadioGroup>

                    <GridList
                        style={{
                            display: 'grid',
                            gap: 'var(--spectrum-global-dimension-size-100)',
                            gridTemplateColumns: '1fr 1fr',
                        }}
                    >
                        <GridListItem textValue='Always'>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='blue-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Checkbox isSelected>Always</Checkbox>
                            </View>
                        </GridListItem>
                        <GridListItem textValue='Confidence'>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='gray-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Checkbox>Confidence (%)</Checkbox>
                            </View>
                        </GridListItem>
                        <GridListItem textValue='Empty'>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='gray-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Checkbox>Empty label</Checkbox>
                            </View>
                        </GridListItem>
                        <GridListItem textValue='Count'>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='gray-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Checkbox>Object count</Checkbox>
                            </View>
                        </GridListItem>
                    </GridList>

                    <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                        Destination
                    </Heading>
                    <RadioGroup defaultValue={'Webhook URL'} label='Output type'>
                        <Grid gap='size-100' columns={['1fr', '1fr']}>
                            <View
                                backgroundColor={'gray-50'}
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                borderColor='gray-500'
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Radio value='MQTT message bus'>MQTT message bus</Radio>
                            </View>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='gray-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Radio value='DDS / ROS2 message bus'>DDS / ROS2 message bus</Radio>
                            </View>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='blue-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Radio value='Webhook URL'>Webhook URL</Radio>
                            </View>
                            <View
                                backgroundColor={'gray-50'}
                                borderColor='gray-500'
                                borderWidth={'thin'}
                                borderRadius={'medium'}
                                padding='size-50'
                                paddingX='size-200'
                            >
                                <Radio value='Folder path'>Folder path</Radio>
                            </View>
                        </Grid>
                    </RadioGroup>
                    <RadioGroup label='Output type' isHidden>
                        <Radio value='MQTT message bus'>MQTT message bus</Radio>
                        <Radio value='DDS / ROS2 message bus'>DDS / ROS2 message bus</Radio>
                        <Radio value='Webhook URL'>Webhook URL</Radio>
                        <Radio value='Folder path'>Folder path</Radio>
                    </RadioGroup>

                    <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                        Predictions format
                    </Heading>

                    <Checkbox>JSON encoded predictions</Checkbox>
                    <Checkbox>Input image with predictions drawn on</Checkbox>
                </Flex>
                <Divider size='S' />
                <Flex justifyContent={'end'}>
                    <ButtonGroup>
                        <Button href={paths.pipeline.model({})} type='button' variant='secondary'>
                            Back
                        </Button>
                        <Button href={paths.liveFeed.index({})} type='submit' variant='accent'>
                            Submit & run
                        </Button>
                    </ButtonGroup>
                </Flex>
            </Flex>
        </Form>
    );
};
