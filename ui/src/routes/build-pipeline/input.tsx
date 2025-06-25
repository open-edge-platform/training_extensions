import { Button, ButtonGroup, Flex, Form, Item, Picker, TextField } from '@geti/ui';

export function BuildInput({ next }: { next: () => void }) {
    return (
        <Flex
            justifyContent={'center'}
            direction='column'
            width={'100'}
            alignContent={'center'}
            alignItems={'center'}
            maxWidth='size-6000'
        >
            <p>
                Please configure the input source for your system. Select the appropriate input type (e.g., IP Camera,
                RTSP stream, etc.) and provide the necessary connection details below.
            </p>
            <Form width='100%'>
                <Picker label='Input source' defaultSelectedKey={'rtsp'}>
                    <Item key='ip-camera'>IP Camera</Item>
                    <Item key='rtsp'>RTSP</Item>
                    <Item key='folder'>Folder</Item>
                </Picker>
                <TextField label='IP Address' />
                <TextField label='Port' />
                <TextField label='Stream Path (RTSP Url path)' />
                <Picker label='Protocol' defaultSelectedKey={'rtsp'}>
                    <Item key='rtsp'>RTSP</Item>
                </Picker>
                <ButtonGroup marginTop={'size-400'}>
                    <Button type='submit' variant='accent' onPress={next}>
                        Save & connect
                    </Button>
                </ButtonGroup>
            </Form>
        </Flex>
    );
}
