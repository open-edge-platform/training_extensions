// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ComponentProps } from 'react';

import { Link as GetiLink } from '@geti/ui';
import { openUrl } from '@tauri-apps/plugin-opener';

export type LinkProps = ComponentProps<typeof GetiLink>;

export const Link = ({ href, target, onPress, ...props }: LinkProps) => {
    const shouldOpenExternally = target === '_blank' && Boolean(href);

    const handlePress: LinkProps['onPress'] = (event) => {
        onPress?.(event);

        if (shouldOpenExternally && href) {
            void openUrl(href);
        }
    };

    return <GetiLink {...props} href={href} target={target} onPress={handlePress} />;
};
