// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { ActionButton, Button, ButtonGroup, Divider, Flex, Form, Text, View } from '@geti/ui';
import { Back } from '@geti/ui/icons';
import { useConnectSourceToPipeline } from 'hooks/api/pipeline.hook';

import type { SourceConfigPayload } from '../../../../constants/shared-types';
import { useSourceAction } from '../hooks/use-source-action.hook';

import classes from './edit-source.module.scss';

interface EditSourceProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    onBackToList: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
    isConnected: boolean;
}

export const EditSource = <T extends SourceConfigPayload>({
    config,
    onSaved,
    onBackToList,
    bodyFormatter,
    componentFields,
    isConnected,
}: EditSourceProps<T>) => {
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: false,
        onSaved: async (sourceId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sourceId));
            connectToPipeline.current = false;
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-100'} alignItems={'center'} marginTop={'0px'}>
                <ActionButton isQuiet onPress={onBackToList}>
                    <Back />
                </ActionButton>

                <Text>Edit input source</Text>
            </Flex>

            <View UNSAFE_className={classes.container}>
                <>{componentFields(state)}</>
            </View>
            <Divider size='S' marginY={'size-200'} />

            <ButtonGroup marginTop={'0px'}>
                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = false)}
                >
                    Save
                </Button>

                {!isConnected && (
                    <Button
                        type='submit'
                        isDisabled={isPending}
                        UNSAFE_style={{ maxWidth: 'fit-content' }}
                        onPress={() => (connectToPipeline.current = true)}
                    >
                        Save & Connect
                    </Button>
                )}
            </ButtonGroup>
        </Form>
    );
};
