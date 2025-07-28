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
    SchemaDdsOutputConfig,
    SchemaDisconnectedOutputConfig,
    SchemaFolderOutputConfig,
    SchemaMqttOutputConfig,
    SchemaRosOutputConfig,
    SchemaWebhookOutputConfig,
} from '../../api/openapi-spec';
import { RadioDisclosure } from '../../components/radio-disclosure-group/radio-disclosure-group';
import { paths } from '../../router';

type OutputConfig =
    | SchemaDisconnectedOutputConfig
    | SchemaDdsOutputConfig
    | SchemaFolderOutputConfig
    | SchemaMqttOutputConfig
    | SchemaWebhookOutputConfig
    | SchemaRosOutputConfig;

type OutputType = OutputConfig['destination_type'];

type ConfigByDestinationType<T extends OutputType> = Extract<OutputConfig, { destination_type: T }>;
type OutputFormRecord = {
    [DestinationTypeKey in OutputType]: ConfigByDestinationType<DestinationTypeKey>;
};
const DEFAULT_OUTPUT_FORMS: OutputFormRecord = {
    disconnected: {
        destination_type: 'disconnected',
        output_formats: [],
    },
    folder: {
        destination_type: 'folder',
        folder_path: '',
        output_formats: [],
    },
    dds: {
        destination_type: 'dds',
        dds_topic: '',
        output_formats: [],
    },
    mqtt: {
        destination_type: 'mqtt',
        broker_host: '',
        broker_port: 1883,
        topic: '',
        output_formats: [],
    },
    ros: {
        destination_type: 'ros',
        ros_topic: '',
        output_formats: [],
    },
    webhook: {
        destination_type: 'webhook',
        webhook_url: '',
        output_formats: [],
    },
};

const ConfigureDisconnectedOutput = ({}: {
    // eslint-disable-next-line react/no-unused-prop-types
    output: SchemaDisconnectedOutputConfig;
    // eslint-disable-next-line react/no-unused-prop-types
    setOutput: (input: SchemaDisconnectedOutputConfig) => void;
}) => {
    return null;
};

const ConfigureFolderOutput = ({
    output,
    setOutput,
}: {
    output: SchemaFolderOutputConfig;
    setOutput: (input: SchemaFolderOutputConfig) => void;
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

const ConfigureDDSOutput = ({
    output,
    setOutput,
}: {
    output: SchemaDdsOutputConfig;
    setOutput: (input: SchemaDdsOutputConfig) => void;
}) => {
    return (
        <TextField
            label='Topic'
            name='dds_topic'
            value={output.dds_topic}
            onChange={(dds_topic) => setOutput({ ...output, dds_topic })}
        />
    );
};

const ConfigureMQTTOutput = ({
    output,
    setOutput,
}: {
    output: SchemaMqttOutputConfig;
    setOutput: (input: SchemaMqttOutputConfig) => void;
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
    output: SchemaRosOutputConfig;
    setOutput: (input: SchemaRosOutputConfig) => void;
}) => {
    return (
        <TextField
            label='Topic'
            name='ros_topic'
            value={output.ros_topic}
            onChange={(ros_topic) => setOutput({ ...output, ros_topic })}
        />
    );
};

const ConfigureWebhookOutput = ({
    output,
    setOutput,
}: {
    output: SchemaWebhookOutputConfig;
    setOutput: (input: SchemaWebhookOutputConfig) => void;
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
    switch (output.destination_type) {
        case 'disconnected':
            return <ConfigureDisconnectedOutput output={output} setOutput={setOutput} />;
        case 'folder':
            return <ConfigureFolderOutput output={output} setOutput={setOutput} />;
        case 'dds':
            return <ConfigureDDSOutput output={output} setOutput={setOutput} />;
        case 'mqtt':
            return <ConfigureMQTTOutput output={output} setOutput={setOutput} />;
        case 'ros':
            return <ConfigureROSOutput output={output} setOutput={setOutput} />;
        case 'webhook':
            return <ConfigureWebhookOutput output={output} setOutput={setOutput} />;
    }
};

const Destinations = ({
    forms,
    setForms,
    selectedDestinationType,
    setSelectedDestinationType,
}: {
    forms: OutputFormRecord;
    setForms: Dispatch<SetStateAction<OutputFormRecord>>;
    selectedDestinationType: OutputType;
    setSelectedDestinationType: Dispatch<SetStateAction<OutputType>>;
}) => {
    return (
        <View>
            <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={2}>
                Destinations
            </Heading>
            <RadioDisclosure
                value={selectedDestinationType}
                setValue={setSelectedDestinationType}
                items={OUTPUT_ITEMS.map((item) => {
                    return {
                        value: item.destination_type,
                        label: <>{item.name}</>,
                        content: (
                            <ConfigureOutput
                                output={forms[item.destination_type]}
                                setOutput={(newOutput) => {
                                    setForms((oldOutput) => {
                                        return { ...oldOutput, [item.destination_type]: newOutput };
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
    { destination_type: 'disconnected', name: 'Disconnected' },
    { destination_type: 'folder', name: 'Folder path' },
    { destination_type: 'dds', name: 'DDS message bus' },
    { destination_type: 'mqtt', name: 'MQTT message bus' },
    { destination_type: 'ros', name: 'ROS2 message bus' },
    { destination_type: 'webhook', name: 'Webhook URL' },
] satisfies Array<{ destination_type: OutputType; name: string }>;

export const Output = () => {
    const navigate = useNavigate();

    const outputs = $api.useSuspenseQuery('get', '/api/outputs');
    const outputMutation = $api.useMutation('post', '/api/outputs');

    const currentOutput: OutputConfig = outputs.data;

    const [selectedDestinationType, setSelectedDestinationType] = useState<OutputType>(currentOutput.destination_type);
    const [forms, setForms] = useState<OutputFormRecord>(() => {
        if (currentOutput !== undefined) {
            return {
                ...DEFAULT_OUTPUT_FORMS,
                [currentOutput.destination_type]: currentOutput,
            };
        }

        return DEFAULT_OUTPUT_FORMS;
    });

    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        outputMutation.mutateAsync({
            body: {
                ...forms[selectedDestinationType],
                output_formats: ['image_original', 'image_with_predictions', 'predictions'] as Array<
                    'image_original' | 'image_with_predictions' | 'predictions'
                >,
                rate_limit: 0.02,
            },
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

                <Destinations
                    forms={forms}
                    setForms={setForms}
                    selectedDestinationType={selectedDestinationType}
                    setSelectedDestinationType={setSelectedDestinationType}
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
