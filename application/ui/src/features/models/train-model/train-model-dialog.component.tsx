// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Link, Text, toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { paths } from '../../../constants/paths';
import { useTrainModelMutation } from '../hooks/api/use-train-model-mutation';
import { AdvancedSettings } from './advanced-settings/advanced-settings.component';
import { BasicTrainModelContent } from './basic-train-model-content.component';
import { TrainModelDialogLayout } from './train-model-dialog-layout';
import { useTrainModel } from './train-model-provider.component';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const { selectedTrainingDevice, selectedModelArchitectureId, selectedDatasetRevision, isValidConfiguration } =
        useTrainModel();
    const trainModelMutation = useTrainModelMutation();
    const projectId = useProjectIdentifier();
    const [isAdvancedSettingsOpen, setIsAdvancedSettingsOpen] = useState<boolean>(false);

    const isStartButtonDisabled = !isValidConfiguration(isAdvancedSettingsOpen);

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
        <Dialog width={'60vw'} height={isAdvancedSettingsOpen ? '80vh' : undefined}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogLayout>
                    {isAdvancedSettingsOpen ? <AdvancedSettings /> : <BasicTrainModelContent />}
                </TrainModelDialogLayout>
            </Content>
            <Divider size={'S'} />
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel
                </Button>
                <Button variant={'secondary'} onPress={() => setIsAdvancedSettingsOpen((prevState) => !prevState)}>
                    {isAdvancedSettingsOpen ? 'Back' : 'Advanced settings'}
                </Button>
                <Button variant={'accent'} onPress={trainModel} isDisabled={isStartButtonDisabled}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
