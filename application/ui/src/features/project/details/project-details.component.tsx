// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Grid, Heading, repeat, Text, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { capitalize, startsWith } from 'lodash-es';
import { Fragment } from 'react/jsx-runtime';

import { $api } from '../../../api/client';
import Background from './../../../assets/background.png';

type FieldProps = {
    field: string;
    value: unknown;
};
const Field = ({ field, value }: FieldProps) => {
    if (!value) {
        return null;
    }

    // 'source_type' => 'Source type'
    const formattedField = capitalize(field.replace(/_/g, ' '));
    const isArrayType = startsWith(`${value}`, '[');

    return (
        <Flex direction={'column'}>
            <Heading level={4}>{formattedField}</Heading>
            <Text>{isArrayType ? `${value}`.replace(/[\[\]']+/g, '') : `${value}`}</Text>
        </Flex>
    );
};

const VerticalHeader = ({ text }: { text: string }) => {
    return (
        <Flex
            alignItems={'center'}
            justifyContent={'center'}
            minWidth={'size-600'}
            UNSAFE_style={{
                writingMode: 'vertical-rl',
                textOrientation: 'mixed',
            }}
        >
            <Heading level={1}>{text}</Heading>
        </Flex>
    );
};

export const ProjectDetails = () => {
    const projectId = useProjectIdentifier();

    const pipeline = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });
    const project = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    const { name, task } = project.data;
    const { task_type, labels = [] } = task;

    return (
        <View
            backgroundColor={'gray-100'}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
                backgroundBlendMode: 'luminosity',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}
            gridArea={'content'}
            height='100%'
            width='100%'
        >
            <View maxWidth={'1048px'} marginX='auto' paddingY='size-800'>
                <View>
                    <Flex direction='column' gap='size-600'>
                        <Flex direction='row' alignItems='stretch' gap='size-200'>
                            <VerticalHeader text={'Project'} />

                            <Divider orientation='vertical' />

                            <Grid columns={repeat(3, '1fr')} rows={repeat(2, 'auto')} gap='size-100' flex='1'>
                                <Heading level={2}>Name</Heading>
                                <Heading level={2}>Task type</Heading>
                                <Heading level={2}>Labels</Heading>

                                <Text>{name}</Text>
                                <Text>{task_type}</Text>
                                <Text>{labels.map((label) => label.name).join(', ')}</Text>
                            </Grid>
                        </Flex>

                        <Divider />

                        <Flex direction='row' alignItems='stretch' gap='size-200'>
                            <VerticalHeader text={'Pipeline'} />

                            <Divider orientation='vertical' />

                            <Grid columns={repeat(3, '1fr')} rows={repeat(5, 'auto')} gap='size-300' flex='1'>
                                <View>
                                    <Heading level={2} marginBottom={'size-300'}>
                                        Source
                                    </Heading>
                                    <Flex direction={'column'} gap={'size-300'}>
                                        {Object.entries(pipeline.data.source || {}).map(
                                            ([field, value], idx, entries) => (
                                                <Fragment key={`source-${field}`}>
                                                    <Field field={field} value={value} />
                                                    {idx < entries.length - 1 && <Divider size={'S'} />}
                                                </Fragment>
                                            )
                                        )}
                                    </Flex>
                                </View>
                                <View>
                                    <Heading level={2} marginBottom={'size-300'}>
                                        Model
                                    </Heading>
                                    <Flex direction={'column'} gap={'size-300'}>
                                        {Object.entries(pipeline.data.model || {}).map(
                                            ([field, value], idx, entries) => {
                                                const displayableFields = ['id', 'architecture'];

                                                if (displayableFields.includes(field)) {
                                                    return (
                                                        <Fragment key={`model-${field}`}>
                                                            <Field field={field} value={value} />
                                                            {idx < entries.length - 1 && <Divider size={'S'} />}
                                                        </Fragment>
                                                    );
                                                }
                                                return null;
                                            }
                                        )}
                                    </Flex>
                                </View>
                                <View>
                                    <Heading level={2} marginBottom={'size-300'}>
                                        Sink
                                    </Heading>
                                    <Flex direction={'column'} gap={'size-300'}>
                                        {Object.entries(pipeline.data.sink || {}).map(
                                            ([field, value], idx, entries) => (
                                                <Fragment key={`sink-${field}`}>
                                                    <Field field={field} value={value} />
                                                    {idx < entries.length - 1 && <Divider size={'S'} />}
                                                </Fragment>
                                            )
                                        )}
                                    </Flex>
                                </View>
                            </Grid>
                        </Flex>
                    </Flex>
                </View>
            </View>
        </View>
    );
};
