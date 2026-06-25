// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, RefObject, useState } from 'react';

import { CustomPopover, dimensionValue, Flex } from '@geti-ui/ui';
import { OverlayTriggerState } from 'react-stately';

import { useEventListener } from '../../hooks/event-listener.hook';

type CursorContextMenuProps = {
    state: OverlayTriggerState;
    children: ReactNode;
    onOpen: () => void;
    triggerRef: RefObject<Element | null>;
};

export const CursorContextMenu = ({ state, children, triggerRef, onOpen }: CursorContextMenuProps) => {
    const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });

    useEventListener(
        'contextmenu',
        (event) => {
            event.preventDefault();
            const modalBox = triggerRef.current?.getBoundingClientRect();

            if (modalBox === undefined) return;

            onOpen();
            setCursorPosition({ x: event.clientX - modalBox.x, y: event.clientY - modalBox.y });
        },
        triggerRef
    );

    return (
        <CustomPopover
            isOpen={state.isOpen}
            onOpenChange={state.setOpen}
            offset={cursorPosition.y}
            crossOffset={cursorPosition.x}
            placement={'bottom start'}
            triggerRef={triggerRef}
        >
            <Flex
                gap={'size-100'}
                justifyContent={'space-between'}
                UNSAFE_style={{ paddingBlock: dimensionValue('size-50') }}
            >
                {children}
            </Flex>
        </CustomPopover>
    );
};
