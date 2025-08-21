import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Grid,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    repeat,
    Tag,
    Text,
    View,
} from '@geti/ui';
import { Calendar, MoreMenu } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import Background from './../../../assets/background.png';
import { NewPipelineButton } from './new-pipeline-button.component';

export const NewPipeline = () => {
    const { data: pipelinesData } = $api.useQuery('get', '/api/pipelines');

    return (
        <DialogTrigger type='fullscreenTakeover'>
            <Button variant='secondary'>New Pipeline</Button>
            {(close) => (
                <Dialog
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-gray-100)',
                        backgroundImage: `url(${Background})`,
                        backgroundBlendMode: 'luminosity',
                        backgroundPosition: 'center',
                        backgroundRepeat: 'no-repeat',
                        backgroundSize: 'cover',
                    }}
                >
                    <Content>
                        <Heading
                            level={1}
                            marginBottom={'size-250'}
                            UNSAFE_style={{
                                textAlign: 'center',
                                fontSize: 'var(--spectrum-global-dimension-font-size-700)',
                            }}
                        >
                            Pipelines
                        </Heading>

                        <Text
                            marginBottom={'size-1000'}
                            UNSAFE_style={{
                                display: 'block',
                                textAlign: 'center',
                                margin: 'auto',
                                maxWidth: '50rem',
                                fontSize: 'var(--spectrum-global-dimension-font-size-200)',
                                lineHeight: 'var(--spectrum-global-dimension-font-size-500)',
                            }}
                        >
                            To create a pipeline, start by defining your objectives and data sources. Then, design the
                            data flow to ensure properprocessing at each stage. Implement the required tools and
                            technologies for automation, and finally,test the pipeline to confirm it runs smoothly and
                            meets your goals.
                        </Text>

                        <Grid columns={repeat('auto-fit', '20rem')} justifyContent='center' gap='size-300'>
                            <NewPipelineButton />

                            {pipelinesData?.map((item, index) => (
                                <View
                                    key={item.id}
                                    padding={'size-275'}
                                    borderWidth={'thin'}
                                    borderColor={'gray-200'}
                                    borderRadius={'regular'}
                                    backgroundColor={'gray-50'}
                                >
                                    <Flex
                                        alignItems={'center'}
                                        marginBottom={'size-200'}
                                        justifyContent={'space-between'}
                                    >
                                        <Flex gap={'size-100'}>
                                            <Heading level={3}>{item.name}</Heading>
                                            {index === 0 && (
                                                <Tag
                                                    withDot={false}
                                                    text='Active'
                                                    style={{
                                                        width: 'fit-content',
                                                        color: 'var(--spectrum-gray-50)',
                                                        background: 'var(--energy-blue)',
                                                        padding: 'var(--spectrum-global-dimension-size-50)',
                                                    }}
                                                />
                                            )}
                                        </Flex>

                                        <MenuTrigger>
                                            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                                                <MoreMenu />
                                            </ActionButton>
                                            <Menu>
                                                <Item>Export</Item>
                                                <Item>Duplicate</Item>
                                                <Item>Delete</Item>
                                            </Menu>
                                        </MenuTrigger>
                                    </Flex>

                                    <Grid
                                        gap={'size-75'}
                                        rows={['auto', 'auto', 'auto']}
                                        columns={['size-675', '1fr']}
                                        marginBottom={'size-200'}
                                    >
                                        <Text>Input:</Text>
                                        <Text>{item.source_id}</Text>
                                        <Text>Model:</Text>
                                        <Text>{item.model_id}</Text>
                                        <Text>Output:</Text>
                                        <Text>{item.sink_id}</Text>
                                    </Grid>

                                    <Flex alignItems={'center'} gap={'size-100'}>
                                        <Calendar width={16} />
                                        <Text
                                            UNSAFE_style={{
                                                color: 'var(--spectrum-gray-700)',
                                                fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                                            }}
                                        >
                                            2025-08-07 06:05 AM
                                        </Text>
                                    </Flex>

                                    <Divider size='S' marginY={'size-200'} />

                                    <Button
                                        variant={'primary'}
                                        marginStart={'auto'}
                                        UNSAFE_style={{
                                            display: 'block',
                                            border: '1px solid var(--spectrum-gray-400)',
                                        }}
                                    >
                                        Run
                                    </Button>
                                </View>
                            ))}
                        </Grid>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            X Close
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
