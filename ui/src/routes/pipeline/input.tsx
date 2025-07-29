import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Divider, Flex, Form, Grid, Loading, NumberField, Text, TextField, View } from '@geti/ui';
import { isEqual } from 'lodash-es';

import { $api } from '../../api/client';
import {
    SchemaDisconnectedSourceConfig,
    SchemaImagesFolderSourceConfig,
    SchemaIpCameraSourceConfig,
    SchemaVideoFileSourceConfig,
    SchemaWebcamSourceConfig,
} from '../../api/openapi-spec';
import { RadioDisclosure } from '../../components/radio-disclosure-group/radio-disclosure-group';
import { Stream } from '../../components/stream/stream';
import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';
import { paths } from '../../router';
import { ReactComponent as Image } from './../../assets/icons/images-folder.svg';
import { ReactComponent as IpCamera } from './../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from './../../assets/icons/video-file.svg';
import { ReactComponent as Webcam } from './../../assets/icons/webcam.svg';

import classes from './../live-feed/live-feed.module.css';

type InputConfig =
    | SchemaDisconnectedSourceConfig
    | SchemaImagesFolderSourceConfig
    | SchemaIpCameraSourceConfig
    | SchemaVideoFileSourceConfig
    | SchemaWebcamSourceConfig;

type SourceType = InputConfig['source_type'];

type ConfigByDestinationType<T extends SourceType> = Extract<InputConfig, { source_type: T }>;
type InputFormRecord = {
    [InputTypeKey in SourceType]: ConfigByDestinationType<InputTypeKey>;
};
const DEFAULT_INPUT_FORMS: InputFormRecord = {
    disconnected: {
        source_type: 'disconnected',
    },
    images_folder: {
        source_type: 'images_folder',
        images_folder_path: '',
        ignore_existing_images: false,
    },
    ip_camera: {
        source_type: 'ip_camera',
        stream_url: '',
        auth_required: false,
    },
    video_file: {
        source_type: 'video_file',
        video_path: '',
    },
    webcam: {
        source_type: 'webcam',
        device_id: 0,
    },
};

const ConnectionPreview = () => {
    const [size, setSize] = useState({ height: 608, width: 892 });
    const { status } = useWebRTCConnection();

    return (
        <>
            {status === 'idle' && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='100%' height='100%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-800)',
                                }}
                            >
                                Save camera settings to establish a connection and view the preview.
                            </Text>
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connecting' && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='100%' height='100%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Loading mode='inline' />
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connected' && (
                <div className={classes.canvasContainer}>
                    <Stream size={size} setSize={setSize} />
                </div>
            )}
        </>
    );
};

const ConfigureDisconnectedInput = ({}: {
    // eslint-disable-next-line react/no-unused-prop-types
    input: SchemaDisconnectedSourceConfig;
    // eslint-disable-next-line react/no-unused-prop-types
    setInput: (input: SchemaDisconnectedSourceConfig) => void;
}) => {
    return null;
};

const ConfigureImagesFolderInput = ({
    input,
    setInput,
}: {
    input: SchemaImagesFolderSourceConfig;
    setInput: (input: SchemaImagesFolderSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Image folder path'
            name='images_folder_path'
            value={input.images_folder_path}
            onChange={(images_folder_path) => setInput({ ...input, images_folder_path })}
        />
    );
};

const ConfigureIpCameraInput = ({
    input,
    setInput,
}: {
    input: SchemaIpCameraSourceConfig;
    setInput: (input: SchemaIpCameraSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Stream URL'
            name='stream_url'
            value={input.stream_url}
            onChange={(stream_url) => setInput({ ...input, stream_url })}
        />
    );
};

const ConfigureVideoFileInput = ({
    input,
    setInput,
}: {
    input: SchemaVideoFileSourceConfig;
    setInput: (input: SchemaVideoFileSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Video file path'
            name='video_path'
            value={input.video_path}
            onChange={(video_path) => setInput({ ...input, video_path })}
        />
    );
};

const ConfigureWebcamInput = ({
    input,
    setInput,
}: {
    input: SchemaWebcamSourceConfig;
    setInput: (input: SchemaWebcamSourceConfig) => void;
}) => {
    return (
        <NumberField
            label='Webcam device id'
            name='device_id'
            hideStepper
            value={input.device_id}
            onChange={(device_id) => setInput({ ...input, device_id })}
        />
    );
};

const ConfigureInput = ({ input, setInput }: { input: InputConfig; setInput: (input: InputConfig) => void }) => {
    switch (input.source_type) {
        case 'disconnected':
            return <ConfigureDisconnectedInput input={input} setInput={setInput} />;
        case 'images_folder':
            return <ConfigureImagesFolderInput input={input} setInput={setInput} />;
        case 'ip_camera':
            return <ConfigureIpCameraInput input={input} setInput={setInput} />;
        case 'video_file':
            return <ConfigureVideoFileInput input={input} setInput={setInput} />;
        case 'webcam':
            return <ConfigureWebcamInput input={input} setInput={setInput} />;
    }
};

const Label = ({ item }: { item: { name: string; source_type: SourceType } }) => {
    return (
        <Flex alignItems='center' gap='size-200'>
            <Flex alignItems='center' justifyContent={'center'}>
                {item.source_type === 'video_file' && <Video width='32px' />}
                {item.source_type === 'webcam' && <Webcam width='32px' />}
                {item.source_type === 'ip_camera' && <IpCamera width='32px' />}
                {item.source_type === 'images_folder' && <Image width='32px' />}
            </Flex>
            {item.name}
        </Flex>
    );
};

const INPUT_ITEMS = [
    { source_type: 'disconnected', name: 'Disconnected' },
    { source_type: 'webcam', name: 'Webcam' },
    { source_type: 'ip_camera', name: 'IP Camera' },
    { source_type: 'video_file', name: 'Video file' },
    { source_type: 'images_folder', name: 'Images folder' },
] satisfies Array<{ source_type: SourceType; name: string }>;

export const Input = () => {
    const { start, status } = useWebRTCConnection();
    const inputs = $api.useSuspenseQuery('get', '/api/inputs');
    const inputMutation = $api.useMutation('post', '/api/inputs', {
        onSuccess: async () => {
            if (status !== 'connected') {
                await start();
            }
        },
    });

    const [selectedSourceType, setSelectedSourceType] = useState<SourceType>(inputs.data.source_type);
    const [forms, setForms] = useState<InputFormRecord>(() => {
        return {
            ...DEFAULT_INPUT_FORMS,
            [inputs.data.source_type]: inputs.data,
        };
    });

    const submitIsDisabled = isEqual(forms[selectedSourceType], inputs.data);
    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        inputMutation.mutateAsync({ body: forms[selectedSourceType] });
    };

    return (
        <Grid
            areas={['text text', 'form canvas', 'divider divider', 'buttons buttons']}
            columns={['auto', '1fr']}
            gap='size-200'
        >
            <View
                gridArea='text'
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                    textAlign: 'center',
                }}
            >
                <Text>
                    Please configure the input source for your system. Select the appropriate input type and provide the
                    necessary connection details below.
                </Text>
            </View>
            <Form gridArea={'form'} onSubmit={onSubmit}>
                <RadioDisclosure
                    ariaLabel={'Select your input source'}
                    value={selectedSourceType}
                    setValue={setSelectedSourceType}
                    items={INPUT_ITEMS.map((item) => {
                        return {
                            value: item.source_type,
                            label: <Label item={item} />,
                            content: (
                                <ConfigureInput
                                    input={forms[item.source_type]}
                                    setInput={(newInput) => {
                                        setForms((oldOutput) => {
                                            return { ...oldOutput, [item.source_type]: newInput };
                                        });
                                    }}
                                />
                            ),
                        };
                    })}
                />

                <ButtonGroup marginTop={'size-400'} marginX='size-200'>
                    <Button
                        type='submit'
                        variant='accent'
                        isPending={inputMutation.isPending}
                        // TODO: disable only if there are no changes
                        isDisabled={submitIsDisabled && status === 'connected'}
                    >
                        Save & connect
                    </Button>
                </ButtonGroup>
            </Form>
            <View gridArea={'canvas'} height={'100%'} UNSAFE_style={{ backgroundColor: 'rgba(0, 0, 0, 0.20)' }}>
                <ConnectionPreview />
            </View>
            <View gridArea={'divider'} paddingY='size-400'>
                <Divider size='S' orientation='horizontal' />
            </View>
            <View gridArea='buttons'>
                <ButtonGroup align={'end'} width={'100%'}>
                    <Button href={paths.pipeline.model({})} variant='accent' isDisabled={status !== 'connected'}>
                        Next
                    </Button>
                </ButtonGroup>
            </View>
        </Grid>
    );
};
