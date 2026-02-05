// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Link, Text, toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useMatch } from 'react-router';

import { paths } from '../../../constants/paths';
import { useTrainModelMutation } from '../hooks/api/use-train-model-mutation';
import { TrainModelDialogContent } from './train-model-dialog-content';
import { useTrainModel } from './train-model-provider.component';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const { selectedTrainingDevice, selectedModelArchitectureId, selectedDatasetRevisionId, datasetRevisions } =
        useTrainModel();
    const trainModelMutation = useTrainModelMutation();
    const projectId = useProjectIdentifier();
    const isModelsPage = useMatch(paths.project.models.pattern);

    const isStartButtonDisabled = selectedModelArchitectureId === null || selectedTrainingDevice === null;

    const trainModel = () => {
        if (isStartButtonDisabled) return;

        const datasetRevisionId = datasetRevisions.find((revision) => revision.id === selectedDatasetRevisionId)?.value;

        trainModelMutation.mutate(
            {
                datasetRevisionId: datasetRevisionId === undefined ? null : datasetRevisionId,
                device: selectedTrainingDevice,
                modelArchitectureId: selectedModelArchitectureId,
            },
            () => {
                onClose();

                toast({
                    message: isModelsPage ? (
                        <Text>Model training started successfully.</Text>
                    ) : (
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
        <Dialog width={'clamp(800px, 50vw, 1150px'}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogContent />
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
