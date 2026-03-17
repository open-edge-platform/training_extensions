// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Footer,
    Heading,
    InlineAlert,
    Link,
    Text,
    toast,
} from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useMatch } from 'react-router';

import { paths } from '../../../constants/paths';
import { useTrainModelMutation } from '../hooks/api/use-train-model-mutation';
import { useIsTrainingButtonDisabled } from '../hooks/use-is-training-button-disabled';
import { AdvancedSettings } from './advanced-settings/advanced-settings.component';
import { BasicTrainModelContent } from './basic-train-model-content.component';
import { TrainModelDialogLayout } from './train-model-dialog-layout.component';
import { useTrainModel } from './train-model-provider.component';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const {
        selectedTrainingDevice,
        selectedModelArchitectureId,
        selectedDatasetRevisionId,
        selectedModelRevisionId,
        datasetRevisions,
        modelRevisions,
        isAdvancedSettingsMode,
        onToggleAdvancedSettingsMode,
        trainingConfiguration,
    } = useTrainModel();
    const trainModelMutation = useTrainModelMutation();
    const projectId = useProjectIdentifier();
    const isModelsPage = useMatch(paths.project.models.pattern);
    const isTrainingDisabled = useIsTrainingButtonDisabled();

    const isStartButtonDisabled =
        isTrainingDisabled || selectedModelArchitectureId === null || selectedTrainingDevice === null;

    const isAdvancedSettingsModeDisabled = selectedModelArchitectureId === null || trainingConfiguration === undefined;

    const trainModel = () => {
        if (isStartButtonDisabled) return;

        const datasetRevisionId = datasetRevisions.find((revision) => revision.id === selectedDatasetRevisionId)?.value;
        const parentModelRevisionId = modelRevisions.find((revision) => revision.id === selectedModelRevisionId)?.value;

        trainModelMutation.mutate(
            {
                datasetRevisionId: datasetRevisionId === undefined ? null : datasetRevisionId,
                parentModelRevisionId: parentModelRevisionId === undefined ? null : parentModelRevisionId,
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
        <Dialog width={'clamp(800px, 50vw, 1150px)'} height={isAdvancedSettingsMode ? '80vh' : undefined}>
            <Heading>Select a model to train</Heading>

            <Divider size={'S'} />

            <Content>
                <TrainModelDialogLayout>
                    {isAdvancedSettingsMode ? <AdvancedSettings /> : <BasicTrainModelContent />}
                </TrainModelDialogLayout>
            </Content>

            <Divider size={'S'} />

            <Footer>
                <Flex alignItems={'center'} marginBottom={'size-200'}>
                    {isTrainingDisabled ? (
                        <InlineAlert variant={'notice'}>
                            <Heading>Why can I not start training?</Heading>
                            <Content>
                                In order to train a model, you need to annotate at least 3 items in your dataset,
                                although we recommend annotating several more for better results.
                            </Content>
                        </InlineAlert>
                    ) : null}
                </Flex>

                <ButtonGroup marginStart={'auto'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        Cancel
                    </Button>
                    {isAdvancedSettingsMode ? (
                        <Button
                            variant={'primary'}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            Back
                        </Button>
                    ) : (
                        <Button
                            variant={'primary'}
                            isDisabled={isAdvancedSettingsModeDisabled}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            Advanced settings
                        </Button>
                    )}

                    <Button variant={'accent'} onPress={trainModel} isDisabled={isStartButtonDisabled}>
                        Start
                    </Button>
                </ButtonGroup>
            </Footer>
        </Dialog>
    );
};
