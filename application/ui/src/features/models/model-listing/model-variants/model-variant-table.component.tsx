// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Cell,
    Column,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    toast,
} from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get } from 'lodash-es';
import { useNumberFormatter } from 'react-aria';

import { API_BASE_URL } from '../../../../api/client';
import type { Model, ModelFormat, ModelVariant } from '../../../../constants/shared-types';
import { downloadFile, formatBytes } from '../../../../shared/util';
import {
    getBaselineVariant,
    getFp32PytorchVariant,
    getPerformanceColumnName,
    getPrimaryTestingMetricValue,
    getVariantPerformanceValue,
} from '../utils/variant-metrics';
import { ModelVariantActions } from './model-variant-actions.component';
import { ValueWithDelta } from './model-variant-delta.component';

type ModelVariantTableProps = {
    model: Model;
    format: ModelFormat;
};

type ModelVariantPrecisionRendererProps = {
    variant: ModelVariant;
};

const ModelVariantPrecisionRenderer = ({ variant }: ModelVariantPrecisionRendererProps) => {
    const numberFormatter = useNumberFormatter({
        style: 'percent',
        maximumFractionDigits: 1,
    });

    if (variant.quantization_info == null) {
        return <Text>{variant.precision.toUpperCase()}</Text>;
    }

    const quantizationParameters = {
        maxDrop: get(variant.quantization_info, 'max_drop', null),
        maxCalibrationSubsetSize: get(variant.quantization_info, 'max_calibration_subset_size', null),
    };

    const maxAccuracyDrop = quantizationParameters.maxDrop === null ? null : Number(quantizationParameters.maxDrop);
    const calibrationDatasetSize =
        quantizationParameters.maxCalibrationSubsetSize === null
            ? null
            : Number(quantizationParameters.maxCalibrationSubsetSize);

    return (
        <Flex direction={'row'} gap={'size-100'}>
            <Text>{variant.precision.toUpperCase()}</Text>
            {(calibrationDatasetSize || maxAccuracyDrop) && (
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>Quantized with NNCF PTQ</Heading>
                    <Content>
                        <Flex direction={'column'}>
                            {maxAccuracyDrop !== null && (
                                <Text>Max accuracy drop: {numberFormatter.format(maxAccuracyDrop)}</Text>
                            )}
                            {calibrationDatasetSize != null && (
                                <Text>Calibration dataset size: {calibrationDatasetSize}</Text>
                            )}
                        </Flex>
                    </Content>
                </ContextualHelp>
            )}
        </Flex>
    );
};

export const ModelVariantTable = ({ model, format }: ModelVariantTableProps) => {
    const projectId = useProjectIdentifier();

    const allVariants = model.variants ?? [];
    const variants = allVariants.filter((variant) => variant.format === format);
    const baselineVariant = getBaselineVariant(variants);
    const fp32PytorchVariant = getFp32PytorchVariant(allVariants);

    const fp32PytorchMetric = getPrimaryTestingMetricValue(fp32PytorchVariant);
    const performanceColumnName = getPerformanceColumnName(variants, fp32PytorchMetric);
    const baselinePerformanceValue = baselineVariant
        ? getVariantPerformanceValue(baselineVariant, fp32PytorchMetric)
        : undefined;

    const handleDownloadModel = (modelVariantId: string) => {
        const url = `${API_BASE_URL}/api/projects/${projectId}/models/${model.id}/variants/${modelVariantId}/binary`;
        downloadFile(url);

        toast({ type: 'info', message: 'Model download started' });
    };

    return (
        <TableView aria-label={`Model variants for ${model.id}`} overflowMode={'wrap'} density={'compact'}>
            <TableHeader>
                <Column isRowHeader>PRECISION</Column>
                <Column isRowHeader>SIZE</Column>
                <Column isRowHeader>{performanceColumnName}</Column>
                <Column align='end'>
                    <></>
                </Column>
            </TableHeader>
            <TableBody items={variants}>
                {(variant) => {
                    const performanceValue = getVariantPerformanceValue(variant, fp32PytorchMetric);
                    const isBaselineVariant = variant.id === baselineVariant?.id;

                    return (
                        <Row key={variant.id}>
                            <Cell>
                                <ModelVariantPrecisionRenderer variant={variant} />
                            </Cell>
                            <Cell>
                                <ValueWithDelta
                                    value={variant.weights_size}
                                    baselineValue={baselineVariant?.weights_size}
                                    changeType='size'
                                    displayValue={formatBytes(variant.weights_size)}
                                    showDelta={!isBaselineVariant}
                                    precision={variant.precision}
                                />
                            </Cell>
                            <Cell>
                                <ValueWithDelta
                                    value={performanceValue}
                                    baselineValue={baselinePerformanceValue}
                                    displayValue={performanceValue === undefined ? '-' : `${performanceValue}%`}
                                    showDelta={!isBaselineVariant}
                                    precision={variant.precision}
                                />
                            </Cell>
                            <Cell>
                                <Flex gap={'size-100'} justifyContent='end' alignItems='center'>
                                    {format !== 'openvino' && (
                                        <ActionButton
                                            isQuiet
                                            aria-label={`Download model ${variant.id}`}
                                            onPress={() => handleDownloadModel(variant.id)}
                                        >
                                            <DownloadIcon />
                                        </ActionButton>
                                    )}
                                    {format === 'openvino' && (
                                        <ModelVariantActions modelVariant={variant} onDownload={handleDownloadModel} />
                                    )}
                                </Flex>
                            </Cell>
                        </Row>
                    );
                }}
            </TableBody>
        </TableView>
    );
};
