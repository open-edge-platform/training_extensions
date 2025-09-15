// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Divider, Flex, Grid, Text } from '@geti/ui';
import { useNavigate } from 'react-router';

import { LabelSelection } from '../../features/project/label-selection/label-selection.component';
import { ProjectName } from '../../features/project/project-name';
import { useProject } from '../../features/project/project-provider.component';
import { TaskSelection } from '../../features/project/task-selection/task-selection.component';
import { paths } from '../../router';
import Background from './../../assets/background.png';

import classes from './project.module.scss';

export const CreateProject = () => {
    const navigate = useNavigate();
    const { createProject } = useProject();

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
                <ProjectName />

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
                <TaskSelection />

                <Divider size='S' />

                <LabelSelection />
            </Flex>

            <Flex justifyContent={'end'} UNSAFE_className={classes.buttonGroup}>
                <ButtonGroup>
                    <Button
                        onPress={() =>
                            createProject({
                                onSuccess: () => {
                                    navigate(paths.inference.index({}));
                                },
                            })
                        }
                        variant='accent'
                    >
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Grid>
    );
};
