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

type OutputConfig =
    | SchemaDisconnectedSinkConfig
    | SchemaFolderSinkConfig
    | SchemaMqttSinkConfig
    | SchemaWebhookSinkConfig
    | SchemaRosSinkConfig;

type OutputType = OutputConfig['sink_type'];

type ConfigBySinkType<T extends OutputType> = Extract<OutputConfig, { sink_type: T }>;
type OutputFormRecord = {
    [SinkTypeKey in OutputType]: ConfigBySinkType<SinkTypeKey>;
};
const DEFAULT_OUTPUT_FORMS: OutputFormRecord = {
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

const ConfigureDisconnectedOutput = ({}: {
    // eslint-disable-next-line react/no-unused-prop-types
    output: SchemaDisconnectedSinkConfig;
    // eslint-disable-next-line react/no-unused-prop-types
    setOutput: (input: SchemaDisconnectedSinkConfig) => void;
}) => {
    return null;
};

const ConfigureFolderOutput = ({
    output,
    setOutput,
}: {
    output: SchemaFolderSinkConfig;
    setOutput: (input: SchemaFolderSinkConfig) => void;
}) => {
    return (
        <TextField
            label='Folder path'
            name='folder_path'
            value={output.folder_path}
            onChange={(folder_path) => setOutput({ ...output, folder_path })}
        />
    );
};

const ConfigureMQTTOutput = ({
    output,
    setOutput,
}: {
    output: SchemaMqttSinkConfig;
    setOutput: (input: SchemaMqttSinkConfig) => void;
}) => {
    return (
        <Flex gap='size-200'>
            <TextField
                label='Broker host'
                name='broker_host'
                value={output.broker_host}
                onChange={(broker_host) => setOutput({ ...output, broker_host })}
            />
            <NumberField
                label='Broker port'
                name='broker_port'
                value={output.broker_port}
                onChange={(broker_port) => setOutput({ ...output, broker_port })}
            />

            <TextField
                label='Topic'
                name='topic'
                value={output.topic}
                onChange={(topic) => setOutput({ ...output, topic })}
            />

            <TextField
                label='Password'
                name='password'
                type='password'
                value={output.password ?? undefined}
                onChange={(password) => setOutput({ ...output, password })}
            />

            <TextField
                label='Username'
                name='username'
                value={output.username ?? undefined}
                onChange={(username) => setOutput({ ...output, username })}
            />
        </Flex>
    );
};

const ConfigureROSOutput = ({
    output,
    setOutput,
}: {
    output: SchemaRosSinkConfig;
    setOutput: (input: SchemaRosSinkConfig) => void;
}) => {
    return (
        <TextField
            label='Topic'
            name='topic'
            value={output.topic}
            onChange={(topic) => setOutput({ ...output, topic })}
        />
    );
};

const ConfigureWebhookOutput = ({
    output,
    setOutput,
}: {
    output: SchemaWebhookSinkConfig;
    setOutput: (input: SchemaWebhookSinkConfig) => void;
}) => {
    return (
        <TextField
            label='Webhook URL'
            name='webhook_url'
            value={output.webhook_url}
            onChange={(webhook_url) => setOutput({ ...output, webhook_url })}
        />
    );
};

const ConfigureOutput = ({
    output,
    setOutput,
}: {
    output: OutputConfig;
    setOutput: (output: OutputConfig) => void;
}) => {
    switch (output.sink_type) {
        case 'disconnected':
            return <ConfigureDisconnectedOutput output={output} setOutput={setOutput} />;
        case 'folder':
            return <ConfigureFolderOutput output={output} setOutput={setOutput} />;
        case 'mqtt':
            return <ConfigureMQTTOutput output={output} setOutput={setOutput} />;
        case 'ros':
            return <ConfigureROSOutput output={output} setOutput={setOutput} />;
        case 'webhook':
            return <ConfigureWebhookOutput output={output} setOutput={setOutput} />;
    }
};

const Sinks = ({
    forms,
    setForms,
    selectedSinkType,
    setSelectedSinkType,
}: {
    forms: OutputFormRecord;
    setForms: Dispatch<SetStateAction<OutputFormRecord>>;
    selectedSinkType: OutputType;
    setSelectedSinkType: Dispatch<SetStateAction<OutputType>>;
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
                items={OUTPUT_ITEMS.map((item) => {
                    return {
                        value: item.sink_type,
                        label: <>{item.name}</>,
                        content: (
                            <ConfigureOutput
                                output={forms[item.sink_type]}
                                setOutput={(newOutput) => {
                                    setForms((oldOutput) => {
                                        return { ...oldOutput, [item.sink_type]: newOutput };
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

const OUTPUT_ITEMS = [
    { sink_type: 'disconnected', name: 'Disconnected' },
    { sink_type: 'folder', name: 'Folder path' },
    { sink_type: 'mqtt', name: 'MQTT message bus' },
    { sink_type: 'ros', name: 'ROS2 message bus' },
    { sink_type: 'webhook', name: 'Webhook URL' },
] satisfies Array<{ sink_type: OutputType; name: string }>;

export const Output = () => {
    const navigate = useNavigate();

    const outputs = $api.useSuspenseQuery('get', '/api/outputs');
    const outputMutation = $api.useMutation('post', '/api/outputs');
    const currentOutput: OutputConfig = outputs.data;

    const [selectedSinkType, setSelectedSinkType] = useState<OutputType>(currentOutput.sink_type);
    const [forms, setForms] = useState<OutputFormRecord>(() => {
        if (currentOutput !== undefined) {
            return {
                ...DEFAULT_OUTPUT_FORMS,
                [currentOutput.sink_type]: currentOutput,
            };
        }

        return DEFAULT_OUTPUT_FORMS;
    });

    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        outputMutation.mutateAsync({
            body: forms[selectedSinkType],
        });

        navigate(paths.liveFeed.index({}));
    };

    return (
        <Form width='100%' onSubmit={onSubmit}>
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
                    <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={2}>
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
                        <Button href={paths.pipeline.model({})} type='button' variant='secondary'>
                            Back
                        </Button>
                        <Button type='submit' variant='accent' isPending={outputMutation.isPending}>
                            Submit & run
                        </Button>
                    </ButtonGroup>
                </Flex>
            </Flex>
        </Form>
    );
};
