import { Button, ButtonGroup, Divider, Flex, Text } from '@geti/ui';

import { LabelSelection } from '../../features/pipelines/models/label-selection.component';
import { ModelSelectionGroup } from '../../features/pipelines/models/model-selection-group.component';
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
                What type of task would you like the model to perform?
            </Text>

            <ModelSelectionGroup />
            <LabelSelection />

            <Divider size='S' />

            <Flex justifyContent={'end'}>
                <ButtonGroup>
                    <Button href={paths.pipeline.input({})} type='submit' variant='secondary'>
                        Back
                    </Button>
                    <Button href={paths.pipeline.output({})} type='submit' variant='accent'>
                        Next
                    </Button>
                </ButtonGroup>
            </Flex>
        </Flex>
    );
};
