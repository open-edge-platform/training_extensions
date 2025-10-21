// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { CSSProperties, Suspense } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, Loading, Text, View } from '@geti/ui';

import { ModelTypesList } from './model-types/model-types-list.component';

import classes from './train-model-dialog.module.scss';

interface TrainModelDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess?: () => void;
}

export const TrainModelDialog = ({ isOpen, onClose, onSuccess }: TrainModelDialogProps) => {
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
                                <ModelTypesList />
                            </Suspense>
                        </View>
                    </Content>
                    <ButtonGroup>
                        <Button variant={'secondary'} onPress={onClose}>
                            Cancel
                        </Button>
                        <Button variant={'accent'} onPress={onClose}>
                            Start
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
