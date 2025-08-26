// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Divider, Flex, Grid, Heading, Text } from '@geti/ui';
import { Edit } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { LabelSelection } from '../../features/project/label-selection.component';
import { ModelSelectionGroup } from '../../features/project/model-selection-group.component';
import { paths } from '../../router';
import Background from './../../assets/background.png';

import classes from './project.module.scss';

export const CreateProject = () => {
    const navigate = useNavigate();

    const handleCreateProject = () => {
        console.info('POST to /models and onSuccess -> Navigate');

        navigate(paths.inference.index({}));
    };

    return (
        <Grid
            UNSAFE_className={classes.grid}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
            }}
            rows={['auto', '1fr', 'auto']}
            height='100%'
            width='100%'
        >
            <Flex
                direction={'column'}
                gap='size-600'
                alignItems={'center'}
                marginTop={'size-1000'}
                marginBottom={'size-400'}
            >
                <Heading level={4}>
                    <Flex alignItems={'center'} gap='size-200'>
                        <Text>Project #1</Text>
                        <Edit fill={'white'} />{' '}
                    </Flex>
                </Heading>

                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-700)',
                        textAlign: 'center',
                    }}
                >
                    What type of task would you like the model to perform?
                </Text>
            </Flex>

            <Flex
                direction='column'
                gap='size-300'
                maxWidth={'1052px'}
                UNSAFE_style={{ overflow: 'auto', margin: '0 auto' }}
            >
                <ModelSelectionGroup />

                <Divider size='S' />

                <LabelSelection />
            </Flex>

            <Flex justifyContent={'end'} UNSAFE_className={classes.buttonGroup}>
                <ButtonGroup>
                    <Button onPress={handleCreateProject} variant='accent'>
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Grid>
    );
};
