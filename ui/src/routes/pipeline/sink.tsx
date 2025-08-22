// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, FormEvent, SetStateAction, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    Divider,
    Flex,
    Form,
    Heading,
    NumberField,
    Text,
    TextField,
    View,
} from '@geti/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../api/client';
import {
    SchemaDisconnectedSinkConfig,
    SchemaFolderSinkConfig,
    SchemaMqttSinkConfig,
    SchemaRosSinkConfig,
    SchemaWebhookSinkConfig,
} from '../../api/openapi-spec';
import { RadioDisclosure } from '../../components/radio-disclosure-group/radio-disclosure-group';
import { paths } from '../../router';
import { ReactComponent as IconFolder } from './../../assets/icons/folder-arrow-right.svg';
import { ReactComponent as IconMQTT } from './../../assets/icons/mqtt.svg';
import { ReactComponent as IconRos } from './../../assets/icons/ros.svg';
import { ReactComponent as IconWebhook } from './../../assets/icons/webhook.svg';

type SinkConfig =
    | SchemaDisconnectedSinkConfig
    | SchemaFolderSinkConfig
    | SchemaMqttSinkConfig
    | SchemaWebhookSinkConfig
    | SchemaRosSinkConfig;

type SinkType = SinkConfig['sink_type'];

type ConfigBySinkType<T extends SinkType> = Extract<SinkConfig, { sink_type: T }>;
type SinkFormRecord = {
    [SinkTypeKey in SinkType]: ConfigBySinkType<SinkTypeKey>;
};
const DEFAULT_SINK_FORMS: SinkFormRecord = {
    disconnected: {
        sink_type: 'disconnected',
        name: 'Disconnected',
        output_formats: [],
    },
    folder: {
        sink_type: 'folder',
        name: 'Folder',
        folder_path: '',
        output_formats: [],
    },
    mqtt: {
        sink_type: 'mqtt',
        name: 'MQTT',
        broker_host: '',
        broker_port: 1883,
        topic: '',
        output_formats: [],
    },
    ros: {
        name: 'Ros',
        sink_type: 'ros',
        topic: '',
        output_formats: [],
    },
    webhook: {
        name: 'Webhook',
        sink_type: 'webhook',
        webhook_url: '',
        output_formats: [],
    },
};

const ConfigureDisconnectedSink = ({}: {
    // eslint-disable-next-line react/no-unused-prop-types
    sink: SchemaDisconnectedSinkConfig;
    // eslint-disable-next-line react/no-unused-prop-types
    setSink: (sink: SchemaDisconnectedSinkConfig) => void;
}) => {
    return null;
};

const ConfigureFolderSink = ({
    sink,
    setSink,
}: {
    sink: SchemaFolderSinkConfig;
    setSink: (sink: SchemaFolderSinkConfig) => void;
}) => {
    return (
        <TextField
            label='Folder path'
            name='folder_path'
            value={sink.folder_path}
            onChange={(folder_path) => setSink({ ...sink, folder_path })}
        />
    );
};

const ConfigureMQTTSink = ({
    sink,
    setSink,
}: {
    sink: SchemaMqttSinkConfig;
    setSink: (sink: SchemaMqttSinkConfig) => void;
}) => {
    return (
        <Flex gap='size-200'>
            <TextField
                label='Broker host'
                name='broker_host'
                value={sink.broker_host}
                onChange={(broker_host) => setSink({ ...sink, broker_host })}
            />
            <NumberField
                label='Broker port'
                name='broker_port'
                value={sink.broker_port}
                onChange={(broker_port) => setSink({ ...sink, broker_port })}
            />

            <TextField
                label='Topic'
                name='topic'
                value={sink.topic}
                onChange={(topic) => setSink({ ...sink, topic })}
            />

            <TextField
                label='Password'
                name='password'
                type='password'
                value={sink.password ?? undefined}
                onChange={(password) => setSink({ ...sink, password })}
            />

            <TextField
                label='Username'
                name='username'
                value={sink.username ?? undefined}
                onChange={(username) => setSink({ ...sink, username })}
            />
        </Flex>
    );
};

const ConfigureROSSink = ({
    sink,
    setSink,
}: {
    sink: SchemaRosSinkConfig;
    setSink: (sink: SchemaRosSinkConfig) => void;
}) => {
    return (
        <TextField label='Topic' name='topic' value={sink.topic} onChange={(topic) => setSink({ ...sink, topic })} />
    );
};

const ConfigureWebhookSink = ({
    sink,
    setSink,
}: {
    sink: SchemaWebhookSinkConfig;
    setSink: (sink: SchemaWebhookSinkConfig) => void;
}) => {
    return (
        <TextField
            label='Webhook URL'
            name='webhook_url'
            value={sink.webhook_url}
            onChange={(webhook_url) => setSink({ ...sink, webhook_url })}
        />
    );
};

const ConfigureSink = ({ sink, setSink }: { sink: SinkConfig; setSink: (sink: SinkConfig) => void }) => {
    switch (sink.sink_type) {
        case 'disconnected':
            return <ConfigureDisconnectedSink sink={sink} setSink={setSink} />;
        case 'folder':
            return <ConfigureFolderSink sink={sink} setSink={setSink} />;
        case 'mqtt':
            return <ConfigureMQTTSink sink={sink} setSink={setSink} />;
        case 'ros':
            return <ConfigureROSSink sink={sink} setSink={setSink} />;
        case 'webhook':
            return <ConfigureWebhookSink sink={sink} setSink={setSink} />;
    }
};

const Sinks = ({
    forms,
    setForms,
    selectedSinkType,
    setSelectedSinkType,
}: {
    forms: SinkFormRecord;
    setForms: Dispatch<SetStateAction<SinkFormRecord>>;
    selectedSinkType: SinkType;
    setSelectedSinkType: Dispatch<SetStateAction<SinkType>>;
}) => {
    return (
        <View>
            <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={2}>
                Sinks
            </Heading>
            <RadioDisclosure
                value={selectedSinkType}
                ariaLabel={'Select the type of sink'}
                setValue={setSelectedSinkType}
                items={SINK_ITEMS.map((item) => {
                    return {
                        value: item.sink_type,
                        label: (
                            <>
                                {item.icon} {item.name}
                            </>
                        ),
                        content: (
                            <ConfigureSink
                                sink={forms[item.sink_type]}
                                setSink={(newSink) => {
                                    setForms((oldSink) => {
                                        return { ...oldSink, [item.sink_type]: newSink };
                                    });
                                }}
                            />
                        ),
                    };
                })}
            />
        </View>
    );
};

const SINK_ITEMS = [
    { sink_type: 'disconnected', name: 'Disconnected', icon: <></> },
    { sink_type: 'folder', name: 'Folder path', icon: <IconFolder /> },
    { sink_type: 'mqtt', name: 'MQTT message bus', icon: <IconMQTT /> },
    { sink_type: 'ros', name: 'ROS2 message bus', icon: <IconRos /> },
    { sink_type: 'webhook', name: 'Webhook URL', icon: <IconWebhook /> },
] satisfies Array<{ sink_type: SinkType; name: string; icon: JSX.Element }>;

export const Sink = () => {
    const navigate = useNavigate();

    const sinks = $api.useSuspenseQuery('get', '/api/sinks');
    const sinkMutation = $api.useMutation('post', '/api/sinks');
    const currentSinks: SinkConfig[] = sinks.data;

    const [selectedSinkType, setSelectedSinkType] = useState<SinkType>(currentSinks[0]?.sink_type ?? 'disconnected');
    const [forms, setForms] = useState<SinkFormRecord>(() => {
        if (currentSinks !== undefined) {
            return {
                ...DEFAULT_SINK_FORMS,
                ...(currentSinks[0] ? { [currentSinks[0].sink_type]: currentSinks[0] } : {}),
            };
        }

        return DEFAULT_SINK_FORMS;
    });

    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        sinkMutation.mutateAsync({
            body: forms[selectedSinkType],
        });

        navigate(paths.inference.index({}));
    };

    return (
        <Form width='100%' onSubmit={onSubmit} maxWidth={'640px'} margin={'0 auto'}>
            <Flex direction='column' gap='size-200'>
                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-700)',
                        textAlign: 'center',
                    }}
                >
                    Define how and where the system should deliver its predictions. This configuration enables seamless
                    integration with your existing workflows and infrastructure.
                </Text>

                <Sinks
                    forms={forms}
                    setForms={setForms}
                    selectedSinkType={selectedSinkType}
                    setSelectedSinkType={setSelectedSinkType}
                />

                <View>
                    <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)', fontWeight: 500 }} level={1}>
                        Predictions format
                    </Heading>

                    <Flex direction='column' gap='size-100'>
                        <Checkbox isEmphasized defaultSelected>
                            JSON encoded predictions
                        </Checkbox>
                        <Checkbox isEmphasized defaultSelected>
                            Input image with predictions drawn on
                        </Checkbox>
                    </Flex>
                </View>

                <Divider size='S' />

                <Flex justifyContent={'end'}>
                    <ButtonGroup>
                        <Button href={paths.pipeline.model({})} variant='secondary'>
                            Back
                        </Button>
                        <Button type='submit' variant='accent' isPending={sinkMutation.isPending}>
                            Submit & run
                        </Button>
                    </ButtonGroup>
                </Flex>
            </Flex>
        </Form>
    );
};
