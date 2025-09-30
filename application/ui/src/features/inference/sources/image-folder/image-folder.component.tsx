// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Switch, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ReactComponent as Folder } from '../../../../assets/icons/folder.svg';
import { ImagesFolderSourceConfig } from '../util';
import { useActionImageFolder } from './use-action-image-folder.hook';

import classes from './image-folder.module.scss';

type ImageFolderProps = {
    config?: ImagesFolderSourceConfig;
};

export const ImageFolder = ({ config }: ImageFolderProps) => {
    const [state, submitAction, isPending] = useActionImageFolder(config, isEmpty(config?.id));

    return (
        <form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <Flex direction='row' gap='size-200'>
                    <TextField
                        flex='1'
                        label='Image folder path'
                        name='images_folder_path'
                        defaultValue={state?.images_folder_path}
                    />

                    <Flex
                        height={'size-400'}
                        alignSelf={'end'}
                        alignItems={'center'}
                        justifyContent={'center'}
                        UNSAFE_className={classes.folderIcon}
                    >
                        <Folder />
                    </Flex>
                </Flex>

                <Switch
                    name='ignore_existing_images'
                    defaultSelected={state?.ignore_existing_images}
                    key={state?.ignore_existing_images ? 'true' : 'false'}
                >
                    Ignore existing images
                </Switch>

                <Button type='submit' maxWidth='size-1000' isDisabled={isPending || isEmpty(state.images_folder_path)}>
                    Apply
                </Button>
            </Flex>
        </form>
    );
};
