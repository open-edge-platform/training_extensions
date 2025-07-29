import { Button, ButtonGroup, Divider, Flex, Grid, Heading, repeat, Text, View } from '@geti/ui';
import { capitalize, isBoolean, isNumber, isObject, isString } from 'lodash-es';

import { $api } from '../../api/client';
import { paths } from '../../router';
import Background from './../../assets/background.png';

interface FieldProps {
    field: string;
    value: unknown;
}
const Field = ({ field, value }: FieldProps) => {
    // 'source_type' => 'Source type'
    const formattedField = capitalize(field.replace(/_/g, ' '));
    const isNonPrimitiveType = !isNumber(value) && !isString(value) && !isBoolean(value);

    if (!value) {
        return null;
    }

    return (
        <Flex direction={'column'}>
            <Heading level={4}>{formattedField}</Heading>
            <Text>{isNonPrimitiveType ? JSON.stringify(value) : value}</Text>
        </Flex>
    );
};

export const Index = () => {
    // TODO: Replace this by /pipeline once available and maybe extract it to a hook
    const inputs = $api.useQuery('get', '/api/inputs');
    const outputs = $api.useQuery('get', '/api/outputs');
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
                                    Input
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {isObject(inputs.data) &&
                                        Object.entries(inputs.data).map(([field, value]) => (
                                            <Field key={field} field={field} value={value} />
                                        ))}
                                </Flex>
                            </View>
                            <View>
                                <Heading level={1} marginBottom={'size-300'}>
                                    Model
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {isObject(models.data) &&
                                        Object.entries(models.data).map(([field, value]) => (
                                            <Field key={field} field={field} value={value} />
                                        ))}
                                </Flex>
                            </View>
                            <View>
                                <Heading level={1} marginBottom={'size-300'}>
                                    Output
                                </Heading>
                                <Flex direction={'column'} gap={'size-300'}>
                                    {isObject(outputs.data) &&
                                        Object.entries(outputs.data).map(([field, value]) => (
                                            <Field key={field} field={field} value={value} />
                                        ))}
                                </Flex>
                            </View>
                        </Grid>
                        <Divider size='S' />
                        <ButtonGroup>
                            <Button href={paths.pipeline.input({})} variant='secondary' marginStart='auto'>
                                Edit
                            </Button>
                        </ButtonGroup>
                    </Flex>
                </View>
            </View>
        </View>
    );
};
