// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    Text,
    toast,
    View,
} from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useSubmitJob } from 'hooks/api/jobs/jobs.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../../api/client';
import {
    CalibrationDatasetSizeField,
    DEFAULT_QUANTIZATION_PARAMETERS,
    MaxAccuracyDropField,
} from './quantization-fields.component';

const useDatasetItemsCount = () => {
    const project_id = useProjectIdentifier();

    const { data: datasetRevisions, isPending } = $api.useQuery('get', '/api/projects/{project_id}/dataset_revisions', {
        params: {
            path: { project_id },
        },
    });

    // Get the latest dataset revision's total item count
    const latestRevision = datasetRevisions?.at(-1);
    const totalCount = latestRevision?.item_counts?.total ?? 0;

    return { totalCount, isPending };
};

type QuantizationDialogProps = {
    modelId: string;
    onClose: () => void;
};

export const QuantizationDialog = ({ modelId, onClose }: QuantizationDialogProps) => {
    const [accuracyDrop, setAccuracyDrop] = useState(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop);
    const [hasNoMaxAccuracyDrop, setHasNoMaxAccuracyDrop] = useState(
        DEFAULT_QUANTIZATION_PARAMETERS.hasNoMaxAccuracyDrop
    );
    const [calibrationSize, setCalibrationSize] = useState(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize);
    const [usesFullCalibrationDataset, setUsesFullCalibrationDataset] = useState(
        DEFAULT_QUANTIZATION_PARAMETERS.usesFullCalibrationDataset
    );

    const { totalCount, isPending: isLoadingCount } = useDatasetItemsCount();
    const submitJob = useSubmitJob();

    const maxCalibrationSize = Math.max(totalCount, 1);
    const effectiveCalibrationSize = Math.min(calibrationSize, maxCalibrationSize);
    const hasNoDatasetItems = !isLoadingCount && totalCount === 0;

    const projectId = useProjectIdentifier();

    const handleStartQuantization = () => {
        submitJob.mutate(
            {
                body: {
                    project_id: projectId,
                    job_type: 'quantize',
                    parameters: {
                        model_id: modelId,
                        max_drop: hasNoMaxAccuracyDrop ? null : accuracyDrop / 100,
                        max_calibration_subset_size: usesFullCalibrationDataset ? totalCount : effectiveCalibrationSize,
                    },
                },
            },
            {
                onSuccess: () => {
                    onClose();
                    toast({
                        type: 'success',
                        message: 'Quantization job started.',
                    });
                },
            }
        );
    };

    return (
        <Dialog width={'100%'}>
            <Heading>Quantization</Heading>

            <Divider size={'S'} />

            <Content>
                <View padding={'size-300'} backgroundColor={'gray-50'} height={'100%'}>
                    <View padding={'size-300'} backgroundColor={'gray-75'} height={'100%'}>
                        <Heading UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }} level={4}>
                            Quantize model to INT8
                        </Heading>

                        <Divider size={'S'} marginY={'size-200'} />

                        <Flex gap={'size-125'} alignItems={'center'} direction={'column'}>
                            <MaxAccuracyDropField
                                value={accuracyDrop}
                                onChange={setAccuracyDrop}
                                isDisabled={hasNoMaxAccuracyDrop}
                                onDisabledChange={setHasNoMaxAccuracyDrop}
                                onReset={() => setAccuracyDrop(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop)}
                            />

                            <CalibrationDatasetSizeField
                                value={effectiveCalibrationSize}
                                onChange={setCalibrationSize}
                                maxValue={maxCalibrationSize}
                                isDisabled={usesFullCalibrationDataset}
                                onDisabledChange={setUsesFullCalibrationDataset}
                                onReset={() => setCalibrationSize(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize)}
                            />
                        </Flex>

                        <Flex gap={'size-100'} alignItems={'center'} marginTop={'size-300'}>
                            <InfoOutline />
                            <Text
                                UNSAFE_style={{
                                    fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                                    color: 'var(--spectrum-global-color-gray-700)',
                                }}
                            >
                                Recommended calibration dataset size: between 200-500 media items
                            </Text>
                        </Flex>
                    </View>
                </View>
            </Content>

            <ButtonGroup>
                <Button
                    variant={'secondary'}
                    onPress={onClose}
                    UNSAFE_style={{ paddingTop: dimensionValue('size-75') }}
                >
                    Cancel
                </Button>
                <Button
                    variant={'primary'}
                    onPress={handleStartQuantization}
                    isPending={submitJob.isPending}
                    isDisabled={isLoadingCount || hasNoDatasetItems || submitJob.isPending}
                >
                    Start quantization
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
