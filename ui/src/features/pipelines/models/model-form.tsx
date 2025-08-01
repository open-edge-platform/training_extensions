import { FormEventHandler, useState } from 'react';

import { Button, ButtonGroup, Flex, Form, TextField, View } from '@geti/ui';

import { $api } from '../../../api/client';
import { DropModelFiles } from './drop-model-files';

type ModelFormData = {
    name: string;
    binFile: null | File;
    xmlFile: null | File;
};

export const UploadModelForm = () => {
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
