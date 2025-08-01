import { Button, ButtonGroup, Divider, Flex, Text } from '@geti/ui';

import { UploadModelForm } from '../../features/pipelines/models/model-form';
import { paths } from '../../router';

export const Model = () => {
    return (
        <Flex direction='column' gap='size-400'>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                    textAlign: 'center',
                }}
            >
                Please upload your trained model to proceed. Ensure the model file is in a supported format and
                compatible with the system.
            </Text>
            <UploadModelForm />
            <Divider size='S' />
            <Flex justifyContent={'end'}>
                <ButtonGroup>
                    <Button href={paths.pipeline.input({})} type='submit' variant='secondary'>
                        Back
                    </Button>
                    <Button href={paths.pipeline.output({})} type='submit' variant='primary'>
                        Next
                    </Button>
                </ButtonGroup>
            </Flex>
        </Flex>
    );
};
