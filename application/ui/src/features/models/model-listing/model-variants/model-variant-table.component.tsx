// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Cell, Column, Flex, Row, TableBody, TableHeader, TableView } from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';

import type { Model, ModelFormat } from '../../../../constants/shared-types';
import { useDownloadModel } from '../../hooks/api/use-download-model.hook';
import { formatModelSize } from '../utils/format-model-size';

interface ModelVariantTableProps {
    model: Model;
    format: ModelFormat;
}

export const ModelVariantTable = ({ model, format }: ModelVariantTableProps) => {
    const { downloadModel, isDownloading } = useDownloadModel(model.id);
    const variants = (model.variants ?? []).filter((variant) => variant.format === format);

    return (
        <TableView aria-label={`Model variants for ${model.id}`} overflowMode={'wrap'} density={'compact'}>
            <TableHeader>
                <Column isRowHeader>PRECISION</Column>
                <Column isRowHeader>SIZE</Column>
                <Column align='end'>
                    <></>
                </Column>
            </TableHeader>
            <TableBody items={variants}>
                {(variant) => (
                    <Row key={`${variant.format}-${variant.precision}`}>
                        <Cell>{variant.precision.toUpperCase()}</Cell>
                        <Cell>{formatModelSize(variant.weights_size)}</Cell>
                        <Cell>
                            <Flex gap={'size-100'} justifyContent='end' alignItems='center'>
                                <ActionButton
                                    isQuiet
                                    aria-label={`Download ${variant.format} model`}
                                    isDisabled={isDownloading}
                                    onPress={() => downloadModel(variant.format as ModelFormat)}
                                >
                                    <DownloadIcon />
                                </ActionButton>
                            </Flex>
                        </Cell>
                    </Row>
                )}
            </TableBody>
        </TableView>
    );
};
