// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, RefObject, useRef, useState } from 'react';

import { View } from '@geti/ui';
import { useOverlay } from 'react-aria';
import { createPortal } from 'react-dom';
import { OverlayTriggerState } from 'react-stately';

import { useEventListener } from '../../hooks/event-listener.hook';

export interface CursorContextMenuProps {
    state: OverlayTriggerState;
    children: ReactNode;
    onOpen: () => void;

    triggerRef: RefObject<Element | null>;
}

export const X_PADDING = 10;

const getParentModal = () => {
    return document.querySelector('[data-testid="modal"]');
};

export const CursorContextMenu = ({ state, children, triggerRef, onOpen }: CursorContextMenuProps) => {
    const ref = useRef<HTMLDivElement>(null);
    const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });

    useEventListener(
        'contextmenu',
        (event) => {
            event.preventDefault();
            const parentModal = getParentModal();

            if (parentModal === null) return;

            const modalBox = parentModal.getBoundingClientRect();

            onOpen();
            setCursorPosition({ x: event.clientX - modalBox.x + X_PADDING, y: event.clientY - modalBox.y });
        },
        triggerRef
    );

    const { overlayProps } = useOverlay(
        {
            isOpen: state.isOpen,
            isDismissable: true,
            shouldCloseOnBlur: false,
            onClose: state.close,
        },
        ref
    );

    if (!state.isOpen) return null;

    return createPortal(
        <div ref={ref} {...overlayProps}>
            <View
                top={cursorPosition.y}
                left={cursorPosition.x}
                zIndex={100001}
                position='absolute'
                data-testid='position container'
                backgroundColor='gray-200'
                {...overlayProps}
            >
                {children}
            </View>
        </div>,
        getParentModal() as HTMLElement
    );
};
