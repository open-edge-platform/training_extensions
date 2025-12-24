// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Button,
    Cell,
    Column,
    Flex,
    Item,
    Menu,
    MenuTrigger,
    Row,
    TableBody,
    TableHeader,
    TableView,
    View,
} from '@geti/ui';
import { DownloadIcon, MoreMenu } from '@geti/ui/icons';

const COLUMN_WIDTHS = {
    name: '2fr' as const,
    license: '1fr' as const,
    precision: '1fr' as const,
    accuracy: '1fr' as const,
    size: '1fr' as const,
    actions: '1fr' as const,
};

const COLUMN_MIN_WIDTHS: Record<string, number> = {
    name: 200,
    license: 120,
    precision: 100,
    accuracy: 100,
    size: 100,
    actions: 200,
};

export const ModelVariantTable = () => {
    // TODO: Replace with dynamic data

    return (
        <View UNSAFE_style={{ overflowX: 'auto' }}>
            <TableView aria-label='Model variants for <variant>' overflowMode={'wrap'} density={'compact'}>
                <TableHeader>
                    <Column isRowHeader width={COLUMN_WIDTHS.name} minWidth={COLUMN_MIN_WIDTHS.name}>
                        Optimized models
                    </Column>
                    <Column isRowHeader width={COLUMN_WIDTHS.license} minWidth={COLUMN_MIN_WIDTHS.license}>
                        License
                    </Column>
                    <Column isRowHeader width={COLUMN_WIDTHS.precision} minWidth={COLUMN_MIN_WIDTHS.precision}>
                        Precision
                    </Column>
                    <Column isRowHeader width={COLUMN_WIDTHS.accuracy} minWidth={COLUMN_MIN_WIDTHS.accuracy}>
                        Accuracy
                    </Column>
                    <Column isRowHeader width={COLUMN_WIDTHS.size} minWidth={COLUMN_MIN_WIDTHS.size}>
                        Size
                    </Column>
                    <Column align='end' width={COLUMN_WIDTHS.actions} minWidth={COLUMN_MIN_WIDTHS.actions}>
                        <></>
                    </Column>
                </TableHeader>
                <TableBody>
                    <Row>
                        <Cell>MobileNetV2-ATSS OpenVINO FP16</Cell>
                        <Cell>Apache 2.0</Cell>
                        <Cell>FP16</Cell>
                        <Cell>95%</Cell>
                        <Cell>335.81 MB</Cell>
                        <Cell>
                            <Flex gap={'size-100'} justifyContent='end' alignItems='center'>
                                <ActionButton isQuiet>
                                    <DownloadIcon />
                                </ActionButton>
                                <MenuTrigger>
                                    <ActionButton isQuiet>
                                        <MoreMenu />
                                    </ActionButton>
                                    <Menu>
                                        <Item key='delete'>Delete</Item>
                                        <Item key='export'>Export</Item>
                                    </Menu>
                                </MenuTrigger>
                            </Flex>
                        </Cell>
                    </Row>
                    <Row>
                        <Cell>MobileNetV2-ATSS OpenVINO FP16</Cell>
                        <Cell>Apache 2.0</Cell>
                        <Cell>FP16</Cell>
                        <Cell>95%</Cell>
                        <Cell>335.81 MB</Cell>
                        <Cell>
                            <Button variant='primary'>Start quantization</Button>
                        </Cell>
                    </Row>
                </TableBody>
            </TableView>
        </View>
    );
};
