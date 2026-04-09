// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Checkbox, Content, Flex, Heading, Text, View } from '@geti/ui';

import { useAcceptLicense } from './api/use-accept-license.hook';

export const License = () => {
    const [isAccepted, setIsAccepted] = useState(false);
    const acceptLicense = useAcceptLicense();

    const handleAccept = () => {
        acceptLicense.mutate({});
    };

    return (
        <View height={'100vh'}>
            <Flex height={'100%'} direction={'column'} alignItems={'center'} justifyContent={'center'} gap={'size-300'}>
                <Heading level={1}>License Agreement</Heading>
                <Content maxWidth={'size-6000'}>
                    <Text>
                        This application uses DINOv3, which is subject to its own license terms. Please review and
                        accept the license before proceeding.{' '}
                        <a
                            href={'https://github.com/IDEA-Research/Grounding-DINO-1.5-API/blob/main/LICENSE'}
                            target={'_blank'}
                            rel={'noopener noreferrer'}
                        >
                            DINOv3 License (Apache 2.0)
                        </a>
                    </Text>
                </Content>
                <Checkbox isSelected={isAccepted} onChange={setIsAccepted}>
                    I have read and accept the third-party license terms
                </Checkbox>
                <Button
                    variant={'accent'}
                    isDisabled={!isAccepted || acceptLicense.isPending}
                    isPending={acceptLicense.isPending}
                    onPress={handleAccept}
                >
                    Accept
                </Button>
            </Flex>
        </View>
    );
};
