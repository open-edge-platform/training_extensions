// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ReactComponent as ONNX } from '../../../../assets/icons/onnx-logo.svg';
import { ReactComponent as OpenVINO } from '../../../../assets/icons/openvino-logo.svg';
import { ReactComponent as Pytorch } from '../../../../assets/icons/pytorch-logo.svg';
import type { ExtendedModel } from '../../../../constants/shared-types';
import { ModelVariantTable } from './model-variant-table.component';

import classes from './model-variant-tabs.module.scss';

type ModelVariantsTabsProps = {
    model: ExtendedModel;
};

export const ModelVariantsTabs = ({ model }: ModelVariantsTabsProps) => {
    if (isEmpty(model.variants)) {
        return (
            <Flex justifyContent={'center'} alignItems={'center'} height={'size-3000'}>
                <Text>There are no model variants available.</Text>
            </Flex>
        );
    }

    return (
        <Tabs aria-label='Model variants' UNSAFE_className={classes.tabs}>
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
            <TabPanels width={0} minWidth={'100%'} UNSAFE_className={classes.tabPanels}>
                <Item key='openvino'>
                    <ModelVariantTable model={model} format='openvino' />
                </Item>
                <Item key='pytorch'>
                    <ModelVariantTable model={model} format='pytorch' />
                </Item>
                <Item key='onnx'>
                    <ModelVariantTable model={model} format='onnx' />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
