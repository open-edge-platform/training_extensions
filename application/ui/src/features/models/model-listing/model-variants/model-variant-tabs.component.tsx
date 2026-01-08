// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, TabList, TabPanels, Tabs } from '@geti/ui';

import { ReactComponent as ONNX } from '../../../../assets/icons/onnx-logo.svg';
import { ReactComponent as OpenVINO } from '../../../../assets/icons/openvino-logo.svg';
import { ReactComponent as Pytorch } from '../../../../assets/icons/pytorch-logo.svg';
import { ModelVariantTable } from './model-variant-table.component';

import classes from './model-variant-tabs.module.scss';

export const ModelVariantsTabs = () => {
    return (
        <Tabs aria-label='Model variants' UNSAFE_className={classes.tabs} marginTop={'size-300'}>
            <TabList>
                <Item key='openvino' textValue='openvino'>
                    <OpenVINO />
                </Item>
                <Item key='pytorch' textValue='pytorch'>
                    <Pytorch />
                </Item>
                <Item key='onnx' textValue='onnx'>
                    <ONNX />
                </Item>
            </TabList>
            <TabPanels width={'calc(100% - 1px)'}>
                <Item key='openvino'>
                    <ModelVariantTable />
                </Item>
                <Item key='pytorch'>Pytorch table here</Item>
                <Item key='onnx'>Onnx table here</Item>
            </TabPanels>
        </Tabs>
    );
};
