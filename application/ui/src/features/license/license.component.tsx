// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, Divider, Flex, Heading, Link, Text, View } from '@geti/ui';

import { useAcceptLicense } from './api/use-accept-license.hook';

export const License = () => {
    const { mutate: acceptLicense, isPending: isAccepting } = useAcceptLicense();

    return (
        <Flex justifyContent={'center'} alignItems={'center'} height={'100vh'}>
            <View
                backgroundColor={'gray-50'}
                padding={'size-400'}
                borderRadius={'regular'}
                maxWidth={'size-6000'}
                width={'100%'}
            >
                <Heading level={2}>License Agreement</Heading>
                <Divider marginY={'size-200'} size={'S'} />
                <Content>
                    <Text>By installing, using, or distributing this application, you acknowledge that:</Text>
                    <ul>
                        <li>you have read and understood the license terms at the links below;</li>
                        <li>confirmed the linked terms govern the contents you seek to access and use; and</li>
                        <li>accepted and agreed to the linked license terms.</li>
                    </ul>
                    <Text>License links</Text>
                    <ul>
                        <li>
                            <Link
                                href={'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md'}
                                target={'_blank'}
                                rel={'noopener noreferrer'}
                            >
                                DINOv3 License
                            </Link>
                        </li>
                    </ul>
                </Content>
                <Flex justifyContent={'end'} marginTop={'size-300'}>
                    <Button
                        variant={'accent'}
                        onPress={() => acceptLicense(undefined)}
                        isPending={isAccepting}
                        isDisabled={isAccepting}
                    >
                        Accept and continue
                    </Button>
                </Flex>
            </View>
        </Flex>
    );
};
