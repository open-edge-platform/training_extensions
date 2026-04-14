// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, Divider, Flex, Heading, Link, Text, View } from '@geti/ui';

import { useAcceptLicense } from './api/use-accept-license.hook';

const LICENSE_LINKS = {
    intelSimplified: {
        label: 'Intel Simplified Software License',
        // eslint-disable-next-line max-len
        href: 'https://www.intel.com/content/www/us/en/content-details/749362/intel-simplified-software-license-version-october-2022.html',
    },
    apache2: {
        label: 'Apache License 2.0',
        href: 'https://www.apache.org/licenses/LICENSE-2.0',
    },
    dinov2: {
        label: 'DINOv2 License (Apache 2.0)',
        href: 'https://github.com/facebookresearch/dinov2/blob/main/LICENSE',
    },
};

type Platform = 'linux' | 'windows' | 'macos';

interface LicenseProps {
    platform: Platform;
}

export const License = ({ platform }: LicenseProps) => {
    const { mutate: acceptLicense, isPending: isAccepting } = useAcceptLicense();

    const appLicense = platform === 'windows' ? LICENSE_LINKS.intelSimplified : LICENSE_LINKS.apache2;

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
                            <Link href={appLicense.href} target={'_blank'} rel={'noopener noreferrer'}>
                                {appLicense.label}
                            </Link>
                        </li>
                        <li>
                            <Link href={LICENSE_LINKS.dinov2.href} target={'_blank'} rel={'noopener noreferrer'}>
                                {LICENSE_LINKS.dinov2.label}
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
