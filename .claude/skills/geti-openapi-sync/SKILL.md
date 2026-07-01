---
name: geti-openapi-sync
description: Regenerate and validate the OpenAPI contract between `application/backend/` and `application/ui/`. Use when backend endpoints, schemas, request or response models, or API surface change, or when `application/ui/src/api/openapi-spec.json` or `openapi-spec.d.ts` is stale. Handles backend spec generation, UI spec placement, TypeScript type regeneration, and the smallest backend and UI checks needed to confirm the contract still matches.
---

# OTX OpenAPI Sync

## Goal

- Keep backend OpenAPI output and UI generated API types in sync.
- Prefer regeneration over manual edits to generated JSON or `.d.ts` files.

## Preferred Workflow

1. If the backend is not already running, generate the spec from `application/backend/` with `just gen-api-spec --output-path ../ui/src/api/openapi-spec.json` on Unix-like shells, or `just gen-api-spec --output-path ..\\ui\\src\\api\\openapi-spec.json` on Windows.
2. If the backend is already running on `http://localhost:7860`, work from `application/ui/` and run `npm run update-spec`.
3. If only the JSON spec changed locally, run `npm run build:api` from `application/ui/` to regenerate `src/api/openapi-spec.d.ts`.
4. Run `npm run format:check` and `npm run type-check` in `application/ui/`, then the narrowest backend or UI tests affected by the contract change.

## When to Use Each Path

- Use direct backend generation when working offline, in CI-like flows, or before the server is runnable.
- Use `npm run update-spec` when actively iterating with a local backend server.
- Use `$geti-backend-dev` for backend fixes if generation exposes schema problems.
- Use `$geti-ui-dev` for UI changes that consume the regenerated types.

## Guardrails

- Commit the generated spec and `.d.ts` together when the contract change is intentional.
- Do not manually edit `application/ui/src/api/openapi-spec.d.ts`.
- If generation fails, fix the backend route or schema definitions instead of patching the generated output.
