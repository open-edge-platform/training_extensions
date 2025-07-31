import { FormEventHandler, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Divider,
    DropZone,
    Flex,
    Form,
    Heading,
    IllustratedMessage,
    Text,
    TextField,
    View,
} from '@geti/ui';
import { Checkmark } from '@geti/ui/icons';
import { isNull } from 'lodash-es';
import { FileTrigger } from 'react-aria-components';

import { $api } from '../../api/client';
import { paths } from '../../router';

//TODO: divide into files!
const DropModelFiles = ({
    files,
    onDrop,
    noFileMessages,
    acceptedFileTypes,
}: {
    files: (File | null)[];
    onDrop: (file: File | null) => void;
    noFileMessages: string[];
    acceptedFileTypes?: string[];
}) => {
    return (
        <DropZone
            flexGrow={1}
            isFilled={files.every((file) => !isNull(file))}
            onDrop={async (dropEvent) => {
                const dropItem = dropEvent.items.at(0) ?? null;

                if (dropItem?.kind === 'file') {
                    onDrop(await dropItem.getFile());
                }
            }}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
            }}
        >
            <IllustratedMessage>
                <Flex direction='column' gap='size-200'>
                    <Heading UNSAFE_style={{ color: 'white' }}>{'Upload your model'}</Heading>

                    <Content>
                        <Flex direction='column' gap='size-400'>
                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-600)',
                                }}
                            >
                                Supported formats: OpenVINO IR (xml + bin)
                            </Text>

                            <Flex direction={'column'}>
                                {files.map((file, index) =>
                                    isNull(file) ? (
                                        <Text
                                            UNSAFE_style={{
                                                color: 'var(--spectrum-global-color-gray-600)',
                                            }}
                                            key={index}
                                        >
                                            {noFileMessages[index] ?? 'You have to drop the file'}
                                        </Text>
                                    ) : (
                                        <Flex key={`${index}-${file.name}`} justifyContent={'center'}>
                                            <Text>{file?.name}</Text>
                                            <Checkmark
                                                size='S'
                                                UNSAFE_style={{ color: 'var(--spectrum-global-color-green-500)' }}
                                            />
                                        </Flex>
                                    )
                                )}
                            </Flex>

                            <FileTrigger
                                acceptedFileTypes={acceptedFileTypes}
                                onSelect={(e) => {
                                    if (e === null) {
                                        return;
                                    }

                                    onDrop(e.item(0));
                                }}
                            >
                                <View>
                                    <Button variant='secondary'>Select files</Button>
                                </View>
                            </FileTrigger>
                        </Flex>
                    </Content>
                </Flex>
            </IllustratedMessage>
        </DropZone>
    );
};

type ModelFormData = {
    name: string;
    binFile: null | File;
    xmlFile: null | File;
};

const UploadModelForm = () => {
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

    const onDropHandler = (file: File | null) => {
        if (file === null) {
            return;
        }
        const type = file.type === 'text/xml' ? 'xmlFile' : file.type === 'application/octet-stream' ? 'binFile' : null;

        if (type === null) {
            return;
        }

        setModelFormData((old) => ({ ...old, [type]: file }));
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

                    <Flex width='100%'>
                        <DropModelFiles
                            files={[modelFormData.xmlFile, modelFormData.binFile]}
                            onDrop={onDropHandler}
                            noFileMessages={['No XML file selected', 'No BIN file selected']}
                            acceptedFileTypes={['text/xml', 'application/octet-stream']}
                        />
                    </Flex>

                    <ButtonGroup align={'end'}>
                        <Button
                            type='submit'
                            isDisabled={!modelFormData.name || !modelFormData.binFile || !modelFormData.xmlFile}
                        >
                            Submit
                        </Button>
                    </ButtonGroup>
                </Flex>
            </View>
        </Form>
    );
};

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
