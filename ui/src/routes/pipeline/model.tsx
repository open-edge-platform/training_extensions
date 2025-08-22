// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Divider, Flex, Text } from '@geti/ui';
import { useNavigate } from 'react-router';

import { LabelSelection } from '../../features/pipelines/label-selection.component';
import { ModelSelectionGroup } from '../../features/pipelines/model-selection-group.component';
import { paths } from '../../router';

export const Model = () => {
    const navigate = useNavigate();

    const handleSubmitSources = () => {
        console.info('POST to /models and onSuccess -> Navigate');

        navigate(paths.inference.index({}));
    };

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
                    <Button variant='secondary'>Back</Button>
                    <Button onPress={handleSubmitSources} variant='primary'>
                        Next
                    </Button>
                </ButtonGroup>
            </Flex>
        </Flex>
    );
};
