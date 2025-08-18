import { Suspense } from 'react';

import { Content, Dialog, DialogTrigger } from '@adobe/react-spectrum';
import { ActionButton, Button, Divider, Flex, Item, Picker } from '@geti/ui';

import { $api } from '../../api/client';
import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';

const DebugTooltip = () => {
    const modelsQuery = $api.useSuspenseQuery('get', '/api/models');

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
                selectedKey={modelsQuery.data[0].name}
                onSelectionChange={(model) => {
                    console.info(model);
                }}
            >
                {modelsQuery.data.map((model) => {
                    return (
                        <Item key={model.id} textValue={model.name}>
                            {model.name}
                        </Item>
                    );
                })}
            </Picker>
        </Flex>
    );
};

export const DebugTrigger = () => {
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
};
