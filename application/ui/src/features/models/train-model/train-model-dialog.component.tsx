// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti/ui';

import { useGetActiveModelArchitectureId } from '../hooks/api/use-get-active-model-architecture-id.hook';
import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingDevices } from '../hooks/api/use-get-training-devices';
import { TrainModelDialogContent } from './train-model-dialog-content';

interface TrainModelDialogProps {
    onClose: () => void;
}

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const { data } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const activeModelArchitectureId = useGetActiveModelArchitectureId();

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(
        activeModelArchitectureId ?? null
    );

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<string | null>(
        trainingDevices?.at(0)?.type ?? null
    );

    const isStartButtonDisabled = selectedModelArchitectureId === null;

    return (
        <Dialog width={'70vw'}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogContent
                    trainingDevices={trainingDevices}
                    selectedTrainingDevice={selectedTrainingDevice}
                    onSelectedTrainingDeviceChange={setSelectedTrainingDevice}
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={data.model_architectures}
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
