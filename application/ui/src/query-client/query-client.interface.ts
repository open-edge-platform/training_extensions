// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { HttpMethod } from 'openapi-typescript-helpers';

import { type paths } from '../api/openapi-spec';

/**
 * Resolves the full operation object for a given path + HTTP method combination
 * from the OpenAPI spec's `paths` map.
 *
 * The OpenAPI spec defines every endpoint as a path entry, each containing one or
 * more HTTP method handlers. For example, `/api/staged_datasets` has both `get`
 * and `post`. `OperationFor` narrows down to the specific method object so that
 * the other helpers can inspect its `parameters`, `requestBody`, and `responses`.
 *
 * @example
 * // Resolves to the full `get_media_annotations_api_...` operation object:
 * type Op = OperationFor<paths, '/api/projects/{project_id}/dataset/media/{media_id}/annotations', 'get'>
 * // => { parameters: { query?: { frame_index?: number | null }; path: { project_id: unknown; media_id: unknown } }; }
 */
type OperationFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> = Method extends keyof Paths[P]
    ? Paths[P][Method]
    : never;

/**
 * Extracts the `path` parameters object from an operation, or `never` if the
 * operation has no path parameters.
 *
 * Path parameters are the dynamic segments embedded in the URL, denoted by
 * curly braces in the OpenAPI spec (e.g. `{project_id}`, `{media_id}`).
 *
 * @example
 * // Has path params → resolves to the path params object:
 * type PP = PathParamsFor<paths, '/api/projects/{project_id}/dataset/media/{media_id}/annotations', 'get'>
 * // => { project_id: string; media_id: string }
 *
 * // Has no path params → resolves to never:
 * type PP = PathParamsFor<paths, '/api/staged_datasets', 'get'>
 * // => never
 */
type PathParamsFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> =
    OperationFor<Paths, P, Method> extends { parameters: { path: infer PP } } ? PP : never;

/**
 * Extracts the `query` parameters object from an operation, or `never` if the
 * operation accepts no query parameters.
 *
 * Query parameters are the key-value pairs appended after `?` in a URL
 * (e.g. `?frame_index=0`). The spec marks the entire `query`
 * block as either required (`query:`) or optional (`query?:`), so both forms
 * are handled here.
 *
 * When the spec writes `query?: never` (explicitly no query params), TypeScript
 * infers `QP` as `never | undefined` = `undefined` in the optional branch.
 * `Exclude<QP, undefined>` strips that back to `never` so those endpoints are
 * treated as having no query params at all.
 *
 * @example
 * // Optional query block (query?:) → resolves to the inner object type (undefined stripped):
 * type QP = QueryParamsFor<paths, '/api/projects/{project_id}/dataset/media/{media_id}/annotations', 'get'>
 * // => { frame_index?: number | null }
 *
 * // Required query block (query:) → resolves to the inner type directly:
 * type QP = QueryParamsFor<paths, '/api/model_architectures', 'get'>
 * // => { task: TaskType }
 *
 * // Explicitly no query params (query?: never) → resolves to never:
 * type QP = QueryParamsFor<paths, '/api/sinks', 'get'>
 * // => never
 *
 * // No query field at all → resolves to never:
 * type QP = QueryParamsFor<paths, '/api/staged_datasets', 'get'>
 * // => never
 */
type QueryParamsFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> =
    OperationFor<Paths, P, Method> extends { parameters: { query: infer QP } }
        ? QP
        : OperationFor<Paths, P, Method> extends { parameters: { query?: infer QP } }
          ? Exclude<QP, undefined>
          : never;

/** Returns all HTTP methods that are actually defined (not `never`) for a given path. */
type MethodsForPath<Paths extends paths, P extends keyof Paths> = Extract<keyof Paths[P], HttpMethod>;

/**
 * Builds the third element of a `QueryKey` tuple — the `{ params: ... }` object —
 * based on which combination of path and query parameters an operation has.
 *
 * The four possible outcomes:
 *
 * 1. **Neither path nor query** → `never`
 *    The `QueryKey` for this operation will be a 2-tuple `[method, path]` with no
 *    params object at all.
 *    @example
 *    // GET /api/staged_datasets — no parameters whatsoever
 *    type K = ['get', '/api/staged_datasets']
 *
 * 2. **Path only** → `{ params: { path: PP } }`
 *    @example
 *    // GET /api/projects/{project_id} — only a project_id in the URL
 *    type K = ['get', '/api/projects/{project_id}', { params: { path: { project_id: unknown } } }]
 *
 * 3. **Query only** → `{ params: { query?: QP } }`
 *    @example
 *    // GET /api/model_architectures?task=detection — only a query string, no path segments
 *    type K = ['get', '/api/model_architectures', { params: { query?: { task: TaskType } } }]
 *
 * 4. **Both path and query** → `{ params: { path: PP; query?: QP } }`
 *    @example
 *    // GET /api/projects/{project_id}/dataset/media/{media_id}/annotations?frame_index=0
 *    type K = [
 *      'get',
 *      '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
 *      { params: { path: { project_id: unknown; media_id: unknown }; query?: { frame_index?: number | null } } }
 *    ]
 *
 * `query` is always marked optional (`query?`) because the spec itself defines the
 * query block as optional (`query?:`) on most operations, meaning callers are never
 * forced to pass it when all query fields are themselves optional.
 */
type ParamsObject<PP, QP> = [PP] extends [never]
    ? [QP] extends [never]
        ? never
        : { params: { query?: QP } }
    : [QP] extends [never]
      ? { params: { path: PP } }
      : { params: { path: PP; query?: QP } };

/**
 * A fully type-safe query key for any endpoint defined in the OpenAPI spec.
 *
 * A `QueryKey` is a tuple whose shape depends on the parameters of the endpoint:
 *
 * - `[method, path]` — for endpoints with no parameters
 * - `[method, path, { params: { path } }]` — for endpoints with only path params
 * - `[method, path, { params: { query? } }]` — for endpoints with only query params
 * - `[method, path, { params: { path, query? } }]` — for endpoints with both
 *
 * TypeScript will enforce the exact shape required by each endpoint, so passing
 * the wrong path params, omitting a required query field, or using the wrong method
 * will all be caught at compile time.
 *
 * Use `getQueryKey()` to construct a validated key at the call site.
 *
 * @example
 * // ✅ No params needed — 2-tuple is enough
 * getQueryKey(['get', '/api/staged_datasets'])
 *
 * // ✅ Path params only
 * getQueryKey(['get', '/api/projects/{project_id}', { params: { path: { project_id: '123' } } }])
 *
 * // ✅ Query params only
 * getQueryKey(['get', '/api/model_architectures', { params: { query: { task: 'detection' } } }])
 *
 * // ✅ Both path and optional query
 * getQueryKey([
 *   'get',
 *   '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
 *   { params: { path: { project_id: '123', media_id: '456' }, query: { frame_index: 0 } } }
 * ])
 *
 * // ❌ Wrong path param key — TypeScript error
 * getQueryKey(['get', '/api/projects/{project_id}', { params: { path: { id: '123' } } }])
 */
export type QueryKey<Paths extends paths> = {
    [P in keyof Paths]: {
        [M in MethodsForPath<Paths, P>]: ParamsObject<
            PathParamsFor<Paths, P, M>,
            QueryParamsFor<Paths, P, M>
        > extends never
            ? [M, P]
            : [M, P, ParamsObject<PathParamsFor<Paths, P, M>, QueryParamsFor<Paths, P, M>>];
    }[MethodsForPath<Paths, P>];
}[keyof Paths];

export type Meta = {
    invalidateQueries?: QueryKey<paths>[];
    awaits?: QueryKey<paths>[];
    error?: {
        message?: string;
        notify?: (error: unknown) => boolean;
    };
};
