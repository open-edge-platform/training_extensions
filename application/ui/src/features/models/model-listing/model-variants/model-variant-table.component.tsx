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
} from '@geti/ui';
import { DownloadIcon, MoreMenu } from '@geti/ui/icons';

const COLUMN_DEFAULT_WIDTH = '1fr';

export const ModelVariantTable = () => {
    // TODO: Replace with dynamic data

    return (
        <TableView aria-label='Model variants for <variant>' overflowMode={'wrap'} density={'compact'}>
            <TableHeader>
                <Column isRowHeader width={'2fr'}>
                    Optimized models
                </Column>
                <Column isRowHeader width={COLUMN_DEFAULT_WIDTH}>
                    License
                </Column>
                <Column isRowHeader width={COLUMN_DEFAULT_WIDTH}>
                    Precision
                </Column>
                <Column isRowHeader width={COLUMN_DEFAULT_WIDTH}>
                    Accuracy
                </Column>
                <Column isRowHeader width={COLUMN_DEFAULT_WIDTH}>
                    Size
                </Column>
                <Column align='end' width={COLUMN_DEFAULT_WIDTH}>
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
                            <MenuTrigger onOpenChange={() => {}}>
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
    );
};
