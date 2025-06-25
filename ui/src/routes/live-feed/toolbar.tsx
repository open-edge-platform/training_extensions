import { StatusLight } from '@adobe/react-spectrum';
import { Divider, Flex, Item, Picker, Text, View } from '@geti/ui';

import { $api } from '../../api/client';

export function Toolbar() {
    const modelsQuery = $api.useQuery('get', '/api/models');
    const activeModelMutation = $api.useMutation('post', '/api/models/{model_name}:activate');

    if (modelsQuery.isLoading || !modelsQuery.data) return 'Loading...';

    if (modelsQuery.error) return `An error occured`;

    return (
        <View
            backgroundColor={'gray-100'}
            gridArea='toolbar'
            padding='size-200'
            UNSAFE_style={{
                fontSize: '12px',
                color: 'var(--spectrum-global-color-gray-800)',
            }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'}>
                <Flex gap='size-50' alignItems='center'>
                    <Picker
                        selectedKey={modelsQuery.data.active_model}
                        onSelectionChange={(model) => {
                            if (model === null || !modelsQuery.data.available_models.some((name) => name === model)) {
                                return;
                            }

                            activeModelMutation.mutate({
                                params: {
                                    path: { model_name: String(model) },
                                },
                            });
                        }}
                    >
                        {modelsQuery.data.available_models.map((model) => {
                            return <Item key={model}>{model}</Item>;
                        })}
                    </Picker>
                    <Text
                        UNSAFE_style={{
                            color: 'var(--spectrum-global-color-gray-900)',
                        }}
                    >
                        MobileNetV2-ATSS
                    </Text>
                    <Text
                        UNSAFE_style={{
                            color: 'var(--spectrum-global-color-gray-700)',
                        }}
                    >
                        Version 2
                    </Text>
                </Flex>
                <Divider orientation='vertical' size='S' />
                <Text>Deployed: 04 March 2021, 3:34 PM</Text>
                <Divider orientation='vertical' size='S' />
                <Flex gap='size-100' alignItems={'center'}>
                    <Text>IP Camera: 192.168.1.100:55</Text>
                    <StatusLight variant='positive' />
                </Flex>
                <Divider orientation='vertical' size='S' />
                <Text>Destination: URL: ../../../</Text>
            </Flex>
        </View>
    );
}
