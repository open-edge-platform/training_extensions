// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, ContextualHelp, Flex, Text } from '@geti/ui';

export const QuantizationRow = () => {
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
            <Button variant={'secondary'}>Start quantization</Button>
        </Flex>
    );
};
