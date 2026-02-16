// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react';

const SCROLL_THRESHOLD = 50;

type UseAutoScrollProps = {
    itemCount: number;
    isDisabled: boolean;
};

export const useAutoScroll = ({ itemCount, isDisabled }: UseAutoScrollProps) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const isUserScrolling = useRef(false);
    const [autoScroll, setAutoScroll] = useState(!isDisabled);

    const scrollToBottom = useCallback(() => {
        const container = scrollRef.current;

        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }, []);

    useEffect(() => {
        if (autoScroll && !isUserScrolling.current) {
            scrollToBottom();
        }
    }, [itemCount, autoScroll, scrollToBottom]);

    const handleScroll = useCallback(() => {
        const container = scrollRef.current;

        if (!container) {
            return;
        }

        const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < SCROLL_THRESHOLD;

        if (isAtBottom) {
            isUserScrolling.current = false;

            if (!isDisabled) {
                setAutoScroll(true);
            }
        } else {
            isUserScrolling.current = true;
            setAutoScroll(false);
        }
    }, [isDisabled]);

    return { scrollRef, autoScroll, setAutoScroll, handleScroll };
};
