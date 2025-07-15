import { FormEventHandler, useState } from 'react';

import { ActionMenu, StatusLight } from '@adobe/react-spectrum';
import {
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    DialogContainer,
    Divider,
    DropZone,
    Flex,
    Form,
    Heading,
    IllustratedMessage,
    Item,
    ListView,
    Text,
    TextField,
    View,
} from '@geti/ui';
import { FileTrigger } from 'react-aria-components';

import { $api } from '../../api/client';
import { paths } from '../../router';

function ListModels() {
    const modelsQuery = $api.useQuery('get', '/api/models');
    const activeModelMutation = $api.useMutation('post', '/api/models/{model_name}:activate');
    const deleteModelMutation = $api.useMutation('delete', '/api/models/{model_name}');

    const [modelToBeDeleted, setModelToBeDeleted] = useState<string | null>(null);

    return (
        <View>
            <DialogContainer onDismiss={() => setModelToBeDeleted(null)}>
                {modelToBeDeleted && (
                    <AlertDialog
                        title='Delete model'
                        variant='warning'
                        primaryActionLabel='Confirm'
                        secondaryActionLabel='Cancel'
                        onPrimaryAction={() => {
                            deleteModelMutation.mutate({
                                params: { path: { model_name: modelToBeDeleted } },
                            });
                        }}
                    >
                        Are you sure you want to delete {modelToBeDeleted}?
                    </AlertDialog>
                )}
            </DialogContainer>

            <ListView
                selectionMode='none'
                aria-label='Available models'
                loadingState={modelsQuery.isLoading ? 'loading' : 'idle'}
            >
                {(modelsQuery.data?.available_models ?? []).map((model) => {
                    const isActive = modelsQuery.data?.active_model === model;

                    return (
                        <Item key={model}>
                            <Flex alignItems={'center'} gap='size-200'>
                                {isActive ? <StatusLight variant='positive'>Active</StatusLight> : null}
                                <Text>{model}</Text>
                            </Flex>
                            <ActionMenu
                                onAction={(key) => {
                                    if (key === 'delete') {
                                        setModelToBeDeleted(model);
                                        return;
                                    }

                                    if (key === 'activate') {
                                        activeModelMutation.mutate({
                                            params: {
                                                path: { model_name: String(model) },
                                            },
                                        });
                                    }
                                }}
                            >
                                <Item key='activate' textValue='Activate'>
                                    Activate
                                </Item>
                                <Item key='delete' textValue='Delete'>
                                    <Text>Delete</Text>
                                </Item>
                            </ActionMenu>
                        </Item>
                    );
                })}
            </ListView>
        </View>
    );
}

function DropDrop({ label, file, onDrop }: { label: string; file: File | null; onDrop: (file: File | null) => void }) {
    const [isFilled, setIsFilled] = useState(false);

    return (
        <DropZone
            flexGrow={1}
            isFilled={isFilled}
            onDrop={async (dropEvent) => {
                const dropItem = dropEvent.items.at(0) ?? null;

                if (dropItem?.kind === 'file') {
                    onDrop(await dropItem.getFile());
                }

                //setFile(null);
                setIsFilled(true);
            }}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
                //borderColor: 'var(--energy-blue)',
            }}
        >
            <IllustratedMessage>
                <Flex direction='column' gap='size-200'>
                    <Heading UNSAFE_style={{ color: 'white' }}>{isFilled ? 'You dropped something!' : label}</Heading>

                    <Content>
                        <Flex direction='column' gap='size-400'>
                            {file === null ? (
                                <Text
                                    UNSAFE_style={{
                                        color: 'var(--spectrum-global-color-gray-600)',
                                    }}
                                >
                                    Supported formats: xml, bin
                                </Text>
                            ) : (
                                <Text>{file.name}</Text>
                            )}

                            <FileTrigger
                                onSelect={(e) => {
                                    if (e === null) {
                                        return;
                                    }

                                    onDrop(e.item(0));
                                }}
                            >
                                <View>
                                    <Button variant='secondary'>Select file</Button>
                                </View>
                            </FileTrigger>
                        </Flex>
                    </Content>
                </Flex>
            </IllustratedMessage>
        </DropZone>
    );
}

type ModelFormData = {
    name: string;
    binFile: null | File;
    xmlFile: null | File;
};

function UplaodModelForm() {
    const [modelFormData, setModelFormData] = useState<ModelFormData>({
        name: '',
        binFile: null,
        xmlFile: null,
    });

    const createModelMutation = $api.useMutation('post', '/api/models');
    const onSubmit: FormEventHandler = (e) => {
        e.preventDefault();

        if (modelFormData.name && modelFormData.binFile && modelFormData.xmlFile) {
            const formData = new FormData();
            formData.append('xml_file', modelFormData.xmlFile);
            formData.append('bin_file', modelFormData.binFile);

            createModelMutation.mutate({
                // @ts-expect-error openapi-typescript does not recognize formData
                body: formData,
                params: { query: { model_name: modelFormData.name } },
            });
        }
    };

    return (
        <Form onSubmit={onSubmit}>
            <View padding='size-400' borderWidth='thin' borderColor={'gray-300'} backgroundColor={'gray-200'}>
                <Flex direction='column' gap='size-200'>
                    <TextField
                        name='Name'
                        label='Model name'
                        isRequired
                        width='size-4600'
                        value={modelFormData.name}
                        onChange={(name) => setModelFormData((old) => ({ ...old, name }))}
                    />

                    <Flex width='100%' gap='size-200'>
                        <DropDrop
                            label='Model .bin file'
                            file={modelFormData.binFile}
                            onDrop={(file: File | null) => {
                                setModelFormData((old) => ({ ...old, binFile: file }));
                            }}
                        />
                        <DropDrop
                            label='Model .xml file'
                            file={modelFormData.xmlFile}
                            onDrop={(file: File | null) => {
                                setModelFormData((old) => ({ ...old, xmlFile: file }));
                            }}
                        />
                    </Flex>

                    <ButtonGroup>
                        <Button type='submit'>Submit</Button>
                    </ButtonGroup>
                </Flex>
            </View>
        </Form>
    );
}

export function Model() {
    return (
        <Flex direction='column' gap='size-400'>
            <UplaodModelForm />
            <Divider size='S' />
            <ListModels />
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
}
