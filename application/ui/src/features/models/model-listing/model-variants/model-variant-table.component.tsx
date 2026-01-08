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

export const ModelVariantTable = () => {
    // TODO: Replace with dynamic data

    return (
        <TableView aria-label='Model variants for <variant>' overflowMode={'wrap'} density={'compact'}>
            <TableHeader>
                <Column isRowHeader>Optimized models</Column>
                <Column isRowHeader>License</Column>
                <Column isRowHeader>Precision</Column>
                <Column isRowHeader>Accuracy</Column>
                <Column isRowHeader>Size</Column>
                <Column align='end'>
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
    );
};
