import { Suspense } from 'react';

import { Content, Dialog, DialogTrigger } from '@adobe/react-spectrum';
import { ActionButton, Button, Divider, Flex, Item, Picker } from '@geti/ui';

import { $api } from '../../api/client';
import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';

function DebugTooltip() {
    const modelsQuery = $api.useSuspenseQuery('get', '/api/models');
    const activeModelMutation = $api.useMutation('post', '/api/models/{model_name}:activate');

    const { start, stop } = useWebRTCConnection();

    return (
        <Flex gap='size-200' direction='column'>
            <Flex gap='size-200'>
                <Button onPress={start}>Start</Button>
                <Button onPress={stop}>Stop</Button>
            </Flex>

            <Divider size='S' />

            <Picker
                label='Select model for inference'
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
                    return (
                        <Item key={model} textValue={model}>
                            {model}
                        </Item>
                    );
                })}
            </Picker>
        </Flex>
    );
}

export function DebugTrigger() {
    return (
        <Suspense fallback={'Loading'}>
            <DialogTrigger type='popover'>
                <ActionButton marginStart={'auto'}>Debug</ActionButton>
                <Dialog>
                    <Content>
                        <DebugTooltip />
                    </Content>
                </Dialog>
            </DialogTrigger>
        </Suspense>
    );
}
