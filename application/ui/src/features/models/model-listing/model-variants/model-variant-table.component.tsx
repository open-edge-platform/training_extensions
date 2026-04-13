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
import { useNumberFormatter } from 'react-aria';

import { API_BASE_URL } from '../../../../api/client';
import type { Model, ModelFormat } from '../../../../constants/shared-types';
import { downloadFile, formatBytes } from '../../../../shared/util';
import {
    getBaselineVariant,
    getFp32PytorchVariant,
    getPerformanceColumnName,
    getPrimaryTestingMetricValue,
    getVariantPerformanceValue,
} from '../utils/variant-metrics';
import { ValueWithDelta } from './model-variant-delta.component';

type ModelVariantTableProps = {
    model: Model;
    format: ModelFormat;
};

type ModelVariant = Model['variants'][number];

const getQuantizationParameter = (
    parameter: 'max_drop' | 'max_calibration_subset_size',
    quantizationParameters: ModelVariant['quantization_info']
) => {
    if (quantizationParameters == null) {
        return null;
    }

    const quantizedParameter = quantizationParameters[parameter];

    return quantizedParameter == null ? null : Number(quantizedParameter);
};

const ModelVariantPrecisionRenderer = ({ variant }: { variant: ModelVariant }) => {
    const numberFormatter = useNumberFormatter({
        style: 'percent',
    });

    if (variant.quantization_info == null) {
        return <Text>{variant.precision.toUpperCase()}</Text>;
    }

    const maxAccuracyDrop = getQuantizationParameter('max_drop', variant.quantization_info);
    const calibrationDatasetSize = getQuantizationParameter('max_calibration_subset_size', variant.quantization_info);

    return (
        <Flex direction={'row'} gap={'size-100'}>
            <Text>{variant.precision.toUpperCase()}</Text>
            <Text>
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>Quantized with NNCF PQT</Heading>
                    <Content>
                        <Flex direction={'column'}>
                            {maxAccuracyDrop && (
                                <Text>Max accuracy drop: {numberFormatter.format(maxAccuracyDrop)}</Text>
                            )}
                            <Text>Calibration dataset size: {calibrationDatasetSize}</Text>
                        </Flex>
                    </Content>
                </ContextualHelp>
            </Text>
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
        toast({ type: 'info', message: 'Model download started...please wait.' });

        const url = `${API_BASE_URL}/api/projects/${projectId}/models/${model.id}/variants/${modelVariantId}/binary`;
        downloadFile(url);
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
                                    <ActionButton
                                        isQuiet
                                        aria-label={`Download model ${variant.id}`}
                                        onPress={() => handleDownloadModel(variant.id)}
                                    >
                                        <DownloadIcon />
                                    </ActionButton>
                                </Flex>
                            </Cell>
                        </Row>
                    );
                }}
            </TableBody>
        </TableView>
    );
};
