# ADR 001: API Client with openapi-fetch

## Context

We needed a type-safe way to communicate with our FastAPI backend. The API contract is defined via OpenAPI spec, and we wanted:

- Full TypeScript type safety
- Auto-generated types from OpenAPI spec
- React Query integration
- Minimal boilerplate
- Runtime validation

## Decision

We use **`openapi-fetch`** with **`openapi-react-query`** for API communication.

### Implementation

```typescript
// src/api/client.ts

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

export const API_BASE_URL = import.meta.env.PUBLIC_API_BASE_URL || '';
export const fetchClient = createFetchClient<paths>({ baseUrl: API_BASE_URL });
export const $api = createClient(fetchClient);
```

### Usage

**Query Example:**

```typescript
const { data, isLoading } = $api.useQuery('get', '/api/projects/{project_id}', {
    params: { path: { project_id: 'abc-123' } },
});
```

**Mutation Example:**

```typescript
const mutation = $api.useMutation('post', '/api/projects');
mutation.mutate({
    body: {
        name: 'New Project',
        task: { task_type: 'detection', labels: [...] }
    }
});
```

### Type Generation

Types are auto-generated from the OpenAPI spec:

```bash
# Fetch latest spec from backend
npm run build:api:download

# Generate TypeScript types
npm run build:api

# Combined command
npm run update-spec
```

This creates `src/api/openapi-spec.d.ts` with full type definitions for all endpoints.

## Alternative Considered

### Axios + Manual Types

- ❌ Requires manual type definitions
- ❌ Types can drift from API
- ✅ Popular, well-known library

## Consequences

### Positive

- ✅ **Full type safety**: Autocomplete for paths, params, body, response
- ✅ **Single source of truth**: OpenAPI spec drives types
- ✅ **React Query integration**: Built-in caching, optimistic updates, mutations
- ✅ **Low maintenance**: Types auto-update when spec changes
- ✅ **Developer experience**: Instant feedback on API changes

### Negative

- ⚠️ **Build step required**: Must run `npm run update-spec` after API changes
- ⚠️ **Backend dependency**: Need running backend to fetch spec
- ⚠️ **Learning curve**: Developers need to understand openapi-fetch patterns

### Neutral

- Type generation is fast (~1 second)
- Generated file is committed to version control (easier CI/CD)

## Implementation Notes

### Proxy Configuration

In development, we use Rsbuild proxy to avoid CORS:

```typescript
// rsbuild.config.ts
server: {
    proxy: {
        '/api': {
            target: 'http://localhost:7860',
            changeOrigin: true,
        },
    },
}
```

### Error Handling

```typescript
const { data, error } = $api.useQuery('get', '/api/projects');

if (error) {
    // error is typed based on OpenAPI error schemas
    console.error(error.message);
}
```

### Invalidation Patterns

```typescript
const mutation = $api.useMutation('post', '/api/projects', {
    meta: {
        invalidateQueries: [['get', '/api/projects']],
    },
});
```

## References

- [openapi-fetch](https://openapi-ts.pages.dev/openapi-fetch/)
- [openapi-react-query](https://openapi-ts.pages.dev/openapi-react-query/)
- [OpenAPI TypeScript](https://openapi-ts.pages.dev/)
