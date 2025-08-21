// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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

type SourceConfig =
    | SchemaDisconnectedSourceConfig
    | SchemaImagesFolderSourceConfig
    | SchemaIpCameraSourceConfig
    | SchemaVideoFileSourceConfig
    | SchemaWebcamSourceConfig;

type SourceType = SourceConfig['source_type'];

type ConfigByDestinationType<T extends SourceType> = Extract<SourceConfig, { source_type: T }>;
type SourceFormRecord = {
    [SourceTypeKey in SourceType]: ConfigByDestinationType<SourceTypeKey>;
};

const DEFAULT_SOURCE_FORMS: SourceFormRecord = {
    disconnected: {
        name: 'Disconnected',
        source_type: 'disconnected',
    },
    images_folder: {
        name: 'Images folder',
        source_type: 'images_folder',
        images_folder_path: '',
        ignore_existing_images: false,
    },
    ip_camera: {
        name: 'Ip camera',
        source_type: 'ip_camera',
        stream_url: '',
        auth_required: false,
    },
    video_file: {
        name: 'Video file',
        source_type: 'video_file',
        video_path: '',
    },
    webcam: {
        name: 'Webcam',
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

const ConfigureDisconnectedSource = (_: {
    source: SchemaDisconnectedSourceConfig;
    setSource: (source: SchemaDisconnectedSourceConfig) => void;
}) => {
    return null;
};

const ConfigureImagesFolderSource = ({
    source,
    setSource,
}: {
    source: SchemaImagesFolderSourceConfig;
    setSource: (source: SchemaImagesFolderSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Image folder path'
            name='images_folder_path'
            value={source.images_folder_path}
            onChange={(images_folder_path) => setSource({ ...source, images_folder_path })}
        />
    );
};

const ConfigureIpCameraSource = ({
    source,
    setSource,
}: {
    source: SchemaIpCameraSourceConfig;
    setSource: (source: SchemaIpCameraSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Stream URL'
            name='stream_url'
            value={source.stream_url}
            onChange={(stream_url) => setSource({ ...source, stream_url })}
        />
    );
};

const ConfigureVideoFileSource = ({
    source,
    setSource,
}: {
    source: SchemaVideoFileSourceConfig;
    setSource: (source: SchemaVideoFileSourceConfig) => void;
}) => {
    return (
        <TextField
            label='Video file path'
            name='video_path'
            value={source.video_path}
            onChange={(video_path) => setSource({ ...source, video_path })}
        />
    );
};

const ConfigureWebcamSource = ({
    source,
    setSource,
}: {
    source: SchemaWebcamSourceConfig;
    setSource: (source: SchemaWebcamSourceConfig) => void;
}) => {
    return (
        <NumberField
            label='Webcam device id'
            name='device_id'
            hideStepper
            value={source.device_id}
            onChange={(device_id) => setSource({ ...source, device_id })}
        />
    );
};

const ConfigureSource = ({
    source,
    setSource,
}: {
    source: SourceConfig;
    setSource: (source: SourceConfig) => void;
}) => {
    switch (source.source_type) {
        case 'disconnected':
            return <ConfigureDisconnectedSource source={source} setSource={setSource} />;
        case 'images_folder':
            return <ConfigureImagesFolderSource source={source} setSource={setSource} />;
        case 'ip_camera':
            return <ConfigureIpCameraSource source={source} setSource={setSource} />;
        case 'video_file':
            return <ConfigureVideoFileSource source={source} setSource={setSource} />;
        case 'webcam':
            return <ConfigureWebcamSource source={source} setSource={setSource} />;
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

const DEFAULT_SOURCE_ITEMS = [
    { source_type: 'disconnected', name: 'Disconnected' },
    { source_type: 'webcam', name: 'Webcam' },
    { source_type: 'ip_camera', name: 'IP Camera' },
    { source_type: 'video_file', name: 'Video file' },
    { source_type: 'images_folder', name: 'Images folder' },
] satisfies Array<{ source_type: SourceType; name: string }>;

export const Source = () => {
    const { status } = useWebRTCConnection();
    const sources = $api.useSuspenseQuery('get', '/api/sources');
    const sourceMutation = $api.useMutation('post', '/api/sources', {
        onSuccess: async () => {
            // TODO: Enable this once WebRTC connection works properly
            // if (status !== 'connected') {
            //     await start();
            // }
        },
    });

    const [selectedSourceType, setSelectedSourceType] = useState<SourceType>(
        sources.data[0]?.source_type ?? DEFAULT_SOURCE_ITEMS[0].source_type
    );
    const [forms, setForms] = useState<SourceFormRecord>(() => {
        return DEFAULT_SOURCE_FORMS;
    });

    const submitIsDisabled = isEqual(forms[selectedSourceType], sources.data);
    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        sourceMutation.mutateAsync({ body: forms[selectedSourceType] });
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
                    Please configure the source for your system. Select the appropriate source type and provide the
                    necessary connection details below.
                </Text>
            </View>
            <Form gridArea={'form'} onSubmit={onSubmit}>
                <RadioDisclosure
                    ariaLabel={'Select your source'}
                    value={selectedSourceType}
                    setValue={setSelectedSourceType}
                    items={DEFAULT_SOURCE_ITEMS.map((item) => {
                        return {
                            value: item.source_type,
                            label: <Label item={item} />,
                            content: (
                                <ConfigureSource
                                    source={forms[item.source_type]}
                                    setSource={(newSource) => {
                                        setForms((oldSource) => {
                                            return { ...oldSource, [item.source_type]: newSource };
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
                        isPending={sourceMutation.isPending}
                        // TODO: disable only if there are no changes
                        isDisabled={submitIsDisabled && status === 'connected'}
                    >
                        Save
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
                    <Button href={paths.pipeline.model({})} variant='accent'>
                        Next
                    </Button>
                </ButtonGroup>
            </View>
        </Grid>
    );
};
