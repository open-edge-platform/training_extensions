// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { CSSProperties, ReactNode } from 'react';

import { Content, ContextualHelp, Text } from '@geti/ui';

interface InfoTooltipProps {
    id?: string;
    tooltipText: ReactNode;
    iconColor?: string | undefined;
    className?: string;
}

export const InfoTooltip = ({ tooltipText, id, iconColor, className }: InfoTooltipProps) => {
    const style = iconColor ? ({ '--spectrum-alias-icon-color': iconColor } as CSSProperties) : {};

    return (
        <ContextualHelp variant='info' id={id} data-testid={id} UNSAFE_className={className} UNSAFE_style={style}>
            <Content marginTop='0'>
                <Text>{tooltipText}</Text>
            </Content>
        </ContextualHelp>
    );
};
