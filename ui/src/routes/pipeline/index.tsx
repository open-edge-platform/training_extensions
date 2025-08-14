import { Button, ButtonGroup, Divider, Flex, Grid, Heading, repeat, Text, View } from '@geti/ui';
import { capitalize, isArray, isObject, startsWith } from 'lodash-es';

import { $api } from '../../api/client';
import { paths } from '../../router';
import Background from './../../assets/background.png';

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

export const Index = () => {
    // TODO: Replace this by /pipeline once available and maybe extract it to a hook
    const sources = $api.useQuery('get', '/api/sources');
    const sinks = $api.useQuery('get', '/api/sinks');
    const models = $api.useQuery('get', '/api/models');

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
                    <Flex direction='column' gap='size-400'>
                        <Grid columns={repeat(3, '1fr')} rows={repeat(5, 'auto')} gap='size-400'>
                            <View>
                                <Heading level={1} marginBottom={'size-300'}>
                                    Source
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {sources.data?.map((item, idx) =>
                                        Object.entries(item).map(([field, value]) => (
                                            <Field key={field + idx} field={field} value={value} />
                                        ))
                                    )}
                                </Flex>
                            </View>
                            <View>
                                <Heading level={1} marginBottom={'size-300'}>
                                    Model
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {isArray(models.data) &&
                                        models.data.map((model) => (
                                            <Field key={model.id} field={model.name} value={model.format} />
                                        ))}
                                </Flex>
                            </View>
                            <View>
                                <Heading level={1} marginBottom={'size-300'}>
                                    Sink
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {sinks.data?.map((item, idx) =>
                                        Object.entries(item).map(([field, value]) => (
                                            <Field key={field + idx} field={field} value={value} />
                                        ))
                                    )}
                                </Flex>
                            </View>
                        </Grid>
                        <Divider size='S' />
                        <ButtonGroup>
                            <Button href={paths.pipeline.source({})} variant='secondary' marginStart='auto'>
                                Edit
                            </Button>
                        </ButtonGroup>
                    </Flex>
                </View>
            </View>
        </View>
    );
};
