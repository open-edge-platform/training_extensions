// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { HttpMethod } from 'openapi-typescript-helpers';

import { type paths } from '../api/openapi-spec';

type OperationFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> = Method extends keyof Paths[P]
    ? Paths[P][Method]
    : never;

type PathParamsFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> =
    OperationFor<Paths, P, Method> extends { parameters: { path: infer PP } } ? PP : never;

type MethodsForPath<Paths extends paths, P extends keyof Paths> = Extract<keyof Paths[P], HttpMethod>;

export type QueryKey<Paths extends paths> = {
    [P in keyof Paths]: {
        [M in MethodsForPath<Paths, P>]: PathParamsFor<Paths, P, M> extends never
            ? [M, P]
            : [
                  M,
                  P,
                  {
                      params: {
                          path: PathParamsFor<Paths, P, M>;
                      };
                  },
              ];
    }[MethodsForPath<Paths, P>];
}[keyof Paths];

export type Meta = {
    invalidateQueries?: QueryKey<paths>[];
    awaits?: QueryKey<paths>[];
    error?: {
        message?: string;
        notify?: boolean;
    };
};
