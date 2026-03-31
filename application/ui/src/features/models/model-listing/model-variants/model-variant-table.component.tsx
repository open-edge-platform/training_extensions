// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Cell, Column, Flex, Row, TableBody, TableHeader, TableView, toast } from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';

import type { Model, ModelFormat, ModelVariant } from '../../../../constants/shared-types';
import { formatBytes } from '../../../../shared/util';
import { useDownloadModel } from '../../hooks/api/use-download-model.hook';
import { getTestingMetrics } from '../components/model-row/utils';

const getPrimaryTestingMetric = (variant: ModelVariant) => {
    return getTestingMetrics(variant.evaluations).find(({ primary }) => primary);
};

const getPrimaryTestingMetricValue = (
    variant: ModelVariant | undefined
): { name: string; value: number } | undefined => {
    const primaryMetric = variant ? getPrimaryTestingMetric(variant) : undefined;

    if (primaryMetric === undefined) {
        return undefined;
    }

    return { name: primaryMetric.name, value: Math.round(primaryMetric.value * 100) };
};

interface ModelVariantTableProps {
    model: Model;
    format: ModelFormat;
}

export const ModelVariantTable = ({ model, format }: ModelVariantTableProps) => {
    const { downloadModel, isDownloading } = useDownloadModel(model.id);
    const allVariants = model.variants ?? [];
    const variants = (model.variants ?? []).filter((variant) => variant.format === format);
    const fp32PytorchVariant = allVariants.find(
        (variant) => variant.format === 'pytorch' && variant.precision === 'fp32'
    );

    const fp32PytorchMetric = getPrimaryTestingMetricValue(fp32PytorchVariant);
    const performanceColumnName =
        variants.map((variant) => getPrimaryTestingMetricValue(variant)).find((metric) => metric !== undefined)?.name ??
        fp32PytorchMetric?.name ??
        'Score';

    const getVariantPerformanceValue = (variant: ModelVariant): number | undefined => {
        const metric = getPrimaryTestingMetricValue(variant);

        if (metric !== undefined) {
            return metric.value;
        }

        if ((variant.evaluations?.length ?? 0) === 0) {
            return fp32PytorchMetric?.value;
        }

        return undefined;
    };

    const handleDownloadModel = (modelVariantId: string) => {
        toast({ type: 'info', message: 'Model download started...please wait.' });
        downloadModel(modelVariantId);
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
                    const performanceValue = getVariantPerformanceValue(variant);

                    return (
                        <Row key={`${variant.format}-${variant.precision}`}>
                            <Cell>{variant.precision.toUpperCase()}</Cell>
                            <Cell>{formatBytes(variant.weights_size)}</Cell>
                            <Cell>{performanceValue === undefined ? '-' : `${performanceValue}%`}</Cell>
                            <Cell>
                                <Flex gap={'size-100'} justifyContent='end' alignItems='center'>
                                    <ActionButton
                                        isQuiet
                                        aria-label={`Download model ${variant.id}`}
                                        isDisabled={isDownloading}
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
