// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Form, NumberField, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';
import { useSinkAction } from '../hooks/use-sink-action.hook';
import { OutputFormats } from '../output-formats.component';
import { LocalFolderSinkConfig, SinkOutputFormats } from '../utils';

import classes from './local-folder.module.scss';

type LocalFolderProps = {
    config?: LocalFolderSinkConfig;
};

const initConfig: LocalFolderSinkConfig = {
    id: 'folder-id',
    name: '',
    sink_type: 'folder',
    rate_limit: 0,
    folder_path: '',
    output_formats: [],
};

export const LocalFolder = ({ config = initConfig }: LocalFolderProps) => {
    const [state, submitAction, isPending] = useSinkAction<LocalFolderSinkConfig>({
        config,
        isNewSink: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            sink_type: 'folder',
            rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
            folder_path: String(formData.get('folder_path')),
            output_formats: formData.getAll('output_formats') as SinkOutputFormats,
        }),
    });

    return (
        <Form action={submitAction}>
            <TextField isHidden label='id' name='id' defaultValue={state?.id} />

            <Flex direction='column' gap='size-200'>
                <Flex direction={'row'} gap='size-200'>
                    <TextField label='Name' name='name' defaultValue={state?.name} />
                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit ?? undefined}
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

                <OutputFormats config={state?.output_formats} />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
