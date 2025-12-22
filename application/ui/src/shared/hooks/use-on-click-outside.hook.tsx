// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect, useRef } from 'react';

export const useOnOutsideClick = <T extends HTMLElement>(ref: RefObject<T | null>, onClickOutside: () => void) => {
    const clickOutsideRef = useRef(onClickOutside);

    useEffect(() => {
        clickOutsideRef.current = onClickOutside;
    }, [onClickOutside]);

    useEffect(() => {
        if (ref.current === null) return;
        const abortController = new AbortController();

        document.addEventListener(
            'click',
            (event) => {
                if (ref.current === null) return;

                if (!ref.current?.contains(event.target as Node)) {
                    clickOutsideRef.current();
                }
            },
            { signal: abortController.signal }
        );
        return () => {
            abortController.abort();
        };
    }, [ref]);
};
