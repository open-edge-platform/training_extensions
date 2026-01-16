// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti/ui';

import { TrainModelDialogContent } from './train-model-dialog-content';

interface TrainModelDialogProps {
    onClose: () => void;
}

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(null);

    const isStartButtonDisabled = selectedModelArchitectureId === null;

    return (
        <Dialog width={'80vw'}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogContent
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={setSelectedModelArchitectureId}
                />
            </Content>
            <Divider size={'S'} />
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel
                </Button>
                <Button variant={'accent'} onPress={onClose} isDisabled={isStartButtonDisabled}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
