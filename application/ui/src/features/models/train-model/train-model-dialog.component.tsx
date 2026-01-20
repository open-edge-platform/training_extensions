// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Link, Text, toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { paths } from '../../../constants/paths';
import { useTrainModelMutation } from '../hooks/api/use-train-model-mutation';
import { TrainModelDialogContent } from './train-model-dialog-content';
import { useTrainModel } from './use-train-model';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const {
        trainingDevices,
        selectedTrainingDevice,
        onSelectedTrainingDeviceChange,
        onSelectedModelArchitectureIdChange,
        selectedModelArchitectureId,
        modelArchitectures,
        selectedDatasetRevision,
        onSelectedDatasetRevisionChange,
        datasetRevisions,
        activeModelArchitectureId,
        isStartButtonDisabled,
    } = useTrainModel();
    const trainModelMutation = useTrainModelMutation();
    const projectId = useProjectIdentifier();

    const trainModel = () => {
        if (selectedTrainingDevice === null || selectedDatasetRevision === null || selectedModelArchitectureId === null)
            return;

        trainModelMutation.mutate(
            {
                device: selectedTrainingDevice,
                datasetRevisionId: selectedDatasetRevision,
                modelArchitectureId: selectedModelArchitectureId,
            },
            () => {
                onClose();

                toast({
                    message: (
                        <Flex alignItems={'center'} gap={'size-50'} wrap={'wrap'}>
                            <Text>
                                Model training started successfully.{' '}
                                <Link href={paths.project.models({ projectId })} UNSAFE_style={{ color: '#fff' }}>
                                    Open models screen to see progress.
                                </Link>
                            </Text>
                        </Flex>
                    ),
                    type: 'success',
                });
            }
        );
    };

    return (
        <Dialog width={'60vw'}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogContent
                    trainingDevices={trainingDevices}
                    selectedTrainingDevice={selectedTrainingDevice}
                    onSelectedTrainingDeviceChange={onSelectedTrainingDeviceChange}
                    datasetRevisions={datasetRevisions}
                    selectedDatasetRevision={selectedDatasetRevision}
                    onSelectedDatasetRevisionChange={onSelectedDatasetRevisionChange}
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
            </Content>
            <Divider size={'S'} />
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel
                </Button>
                <Button variant={'accent'} onPress={trainModel} isDisabled={isStartButtonDisabled}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
