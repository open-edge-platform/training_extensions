// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, ContextualHelp, DialogTrigger, Flex, Text } from '@geti/ui';

import { QuantizationDialog } from './quantization-dialog/quantization-dialog.component';

type QuantizationRowProps = {
    modelId: string;
};
export const QuantizationRow = ({ modelId }: QuantizationRowProps) => {
    return (
        <Flex marginTop={'size-150'} alignItems={'center'} justifyContent={'space-between'}>
            <Flex>
                <Text>Optimize the FP16 model using OpenVINO NNCF (via INT8 quantization)</Text>
                <ContextualHelp>
                    <Content>
                        OpenVINO NNCF (Neural Network Compression Framework) via INT8 quantization reduces model size
                        and speeds up inference with minimal impact on accuracy
                    </Content>
                </ContextualHelp>
            </Flex>
            <DialogTrigger>
                <Button variant={'secondary'}>Start quantization</Button>
                {(close) => <QuantizationDialog modelId={modelId} onClose={close} />}
            </DialogTrigger>
        </Flex>
    );
};
