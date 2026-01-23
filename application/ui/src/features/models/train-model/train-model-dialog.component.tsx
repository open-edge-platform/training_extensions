// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Link, Text, toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { paths } from '../../../constants/paths';
import { AdvancedSettings } from './advanced-settings/advanced-settings.component';
import { BasicTrainModelContent } from './basic-train-model-content.component';
import { TrainModelDialogLayout } from './train-model-dialog-layout';
import { useTrainModelState } from './train-model-provider.component';
import { useTrainModel } from './use-train-model';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const { isAdvancedSettingsMode, onAdvancedSettingsModeChange, isValidConfiguration } = useTrainModelState();
    const projectId = useProjectIdentifier();

    const { trainModel, isPending } = useTrainModel();

    const isStartButtonDisabled = !isValidConfiguration() || isPending;

    const handleTrainModel = () => {
        trainModel(() => {
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
        });
    };

    return (
        <Dialog width={'60vw'} height={isAdvancedSettingsMode ? '80vh' : undefined}>
            <Heading>Select a model to train</Heading>
            <Divider size={'S'} />
            <Content>
                <TrainModelDialogLayout>
                    {isAdvancedSettingsMode ? <AdvancedSettings /> : <BasicTrainModelContent />}
                </TrainModelDialogLayout>
            </Content>
            <Divider size={'S'} />
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel
                </Button>
                <Button variant={'secondary'} onPress={() => onAdvancedSettingsModeChange(!isAdvancedSettingsMode)}>
                    {isAdvancedSettingsMode ? 'Back' : 'Advanced settings'}
                </Button>
                <Button variant={'accent'} onPress={handleTrainModel} isDisabled={isStartButtonDisabled}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
