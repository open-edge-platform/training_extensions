// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, Loading, Text, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { $api } from 'src/api/client';

import { ModelTypesList } from './model-types-list/model-types-list.component';

import classes from './train-model-dialog.module.scss';

interface TrainModelDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

export const TrainModelDialog = ({ isOpen, onClose }: TrainModelDialogProps) => {
    const project_id = useProjectIdentifier();
    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(null);
    const jobMutation = $api.useMutation('post', '/api/jobs');

    const handleSubmit = async () => {
        await jobMutation.mutateAsync({
            body: {
                job_type: 'train',
                project_id,
                parameters: {
                    model_architecture_id: String(selectedModelArchitectureId),
                },
            },
        });
        onClose();
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <Dialog maxWidth={'100rem'} width={'80vw'}>
                    <Heading>
                        <Text UNSAFE_className={classes.title}>Train model</Text>
                    </Heading>
                    <Divider marginBottom={'size-100'} />
                    <Content minHeight={'size-1000'}>
                        <View
                            flex={1}
                            padding={'size-250'}
                            minHeight={0}
                            backgroundColor={'gray-50'}
                            overflow={'hidden auto'}
                        >
                            <Text UNSAFE_className={classes.subtitle} marginBottom={'size-100'}>
                                Model type
                            </Text>
                            <Suspense fallback={<Loading mode='inline' />}>
                                <ModelTypesList
                                    selectedModelArchitectureId={selectedModelArchitectureId}
                                    setSelectedModelArchitectureId={setSelectedModelArchitectureId}
                                />
                            </Suspense>
                        </View>
                    </Content>
                    <ButtonGroup>
                        <Button variant={'secondary'} onPress={onClose}>
                            Cancel
                        </Button>
                        <Button
                            variant={'accent'}
                            onPress={handleSubmit}
                            isPending={jobMutation.isPending}
                            isDisabled={isEmpty(selectedModelArchitectureId) || jobMutation.isPending}
                        >
                            Start
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
