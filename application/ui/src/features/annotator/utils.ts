// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { PointerEvent, SVGProps } from 'react';

import { isFunction, isNil, negate } from 'lodash-es';

import { isLeftButton, isWheelButton } from '../../shared/buttons-utils';

type OnPointerDown = SVGProps<SVGElement>['onPointerDown'];
export const allowPanning = (onPointerDown?: OnPointerDown): OnPointerDown | undefined => {
    if (onPointerDown === undefined) {
        return;
    }

    return (event: PointerEvent<SVGElement>) => {
        const isPressingPanningHotKeys = (isLeftButton(event) && event.ctrlKey) || isWheelButton(event);

        if (isPressingPanningHotKeys) {
            return;
        }

        return onPointerDown(event);
    };
};

export const DEFAULT_ANNOTATION_STYLES = {
    fillOpacity: 'var(--annotation-fill-opacity)',
    fill: 'var(--annotation-fill)',
    stroke: 'var(--annotation-stroke)',
    strokeLinecap: 'round',
    strokeWidth: 'calc(2px / var(--zoom-scale))',
    strokeDashoffset: 0,
    strokeDasharray: 0,
    strokeOpacity: 'var(--annotation-border-opacity, 1)',
} satisfies SVGProps<SVGElement>;

export const runWhen =
    <T>(predicate: (...args: T[]) => boolean) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (whenTrueFn: (...args: any[]) => void, whenFalseFn?: (...args: any[]) => void) =>
    (...args: T[]): void => {
        if (predicate(...args)) {
            whenTrueFn(...args);
        } else {
            isFunction(whenFalseFn) && whenFalseFn(...args);
        }
    };

export const runWhenTruthy = runWhen(negate(isNil));
