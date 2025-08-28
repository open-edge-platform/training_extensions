// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, DialogTrigger, Grid, Heading, Text } from '@geti/ui';
import { CloseSmall } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { mockedProjects } from './mocked-projects';
import { NewProjectLink } from './new-project-link.component';
import { ProjectCard } from './project-card';

import classes from './project-list.module.scss';

export const ProjectList = () => {
    return (
        <DialogTrigger type='fullscreenTakeover'>
            <Button variant='secondary'>New Project</Button>
            {(close) => (
                <Dialog UNSAFE_className={classes.dialog}>
                    <Content>
                        <Heading
                            level={1}
                            marginBottom={'size-250'}
                            UNSAFE_style={{
                                textAlign: 'center',
                                fontSize: 'var(--spectrum-global-dimension-font-size-700)',
                            }}
                        >
                            Projects
                        </Heading>

                        <Text UNSAFE_className={classes.description}>
                            To create a project, start by defining your objectives. Then, design the data flow to ensure
                            proper processing at each stage. Implement the required tools and technologies for
                            automation, and finally, test the project to confirm it runs smoothly and meets your goals.
                        </Text>

                        <Grid
                            gap={'size-300'}
                            marginX={'auto'}
                            justifyContent={'center'}
                            columns={isEmpty(mockedProjects) ? ['size-3600'] : ['1fr', '1fr']}
                        >
                            <NewProjectLink />

                            {mockedProjects?.map((item, index) => (
                                <ProjectCard key={item.id} item={item} isActive={index === 0} />
                            ))}
                        </Grid>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close} UNSAFE_className={classes.button}>
                            <CloseSmall /> Close
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
