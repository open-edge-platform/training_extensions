// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { Button, Flex, Form, NumberField, TextField } from '@geti/ui';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';
import { OutputFormats } from '../output-formats.component';
import { OutputFormat, SinkType } from '../utils';

import classes from './local-folder.module.scss';

type FolderFormData = {
    name: string;
    sink_type: SinkType;
    rate_limit: number;
    folder_path: string;
    output_formats: OutputFormat[];
};

export const LocalFolder = () => {
    const initData = {
        name: '124',
        sink_type: SinkType.FOLDER,
        rate_limit: 0.2,
        folder_path: './123',
        output_formats: [OutputFormat.IMAGE_ORIGINAL, OutputFormat.PREDICTIONS],
    };

    const [state, submitAction, isPending] = useActionState<FolderFormData, FormData>(async (_prevState, formData) => {
        //Todo: call create endpoint
        return {
            name: formData.get('name'),
            sink_type: SinkType.FOLDER,
            rate_limit: formData.get('rate_limit'),
            folder_path: formData.get('folder_path'),
            output_formats: formData.getAll('output_formats'),
        } as unknown as FolderFormData;
    }, initData);

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <Flex direction={'row'} gap='size-200'>
                    <TextField label='Name' name='name' defaultValue={state?.name} />
                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit}
                    />
                </Flex>

                <Flex direction='row' gap='size-200'>
                    <TextField
                        width={'100%'}
                        label='Folder Path'
                        name='folder_path'
                        defaultValue={state?.folder_path}
                    />

                    <Flex
                        alignSelf={'end'}
                        height={'size-400'}
                        alignItems={'center'}
                        justifyContent={'center'}
                        UNSAFE_className={classes.folderIcon}
                    >
                        <Folder />
                    </Flex>
                </Flex>

                <OutputFormats />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
