---
name: otx-ui-dev
description: Develop and validate changes in `application/ui/` for the React and TypeScript frontend. Use when touching `application/ui/src/**`, frontend tests, RSBuild or Vitest config, Playwright setup, package scripts, or generated API typings under `src/api`. Helps with Node and npm requirements, install and build commands, lint, typecheck, test workflows, and coordination with backend OpenAPI changes.
---

# OTX UI Development

## Quick Start

- Work from `application/ui/`.
- Use Node `>=v24.2.0` and npm `>=11.3.0`.
- Install or refresh dependencies with `npm ci`.
- Start with `npm run format:check`, `npm run lint`, `npm run cyclic-deps-check`, and `npm run type-check`.

## Workflow

1. Keep the change inside the existing UI structure under `src/` unless the task explicitly calls for build or tooling updates.
2. Use existing component, routing, testing, and styling patterns instead of introducing a new structure.
3. Regenerate API types instead of hand-editing them when the backend contract changes.
4. Escalate to component or e2e tests only when the change affects rendered browser behavior.

## Verification

- Use `npm run format:check` for Prettier verification.
- Use `npm run lint`, `npm run cyclic-deps-check`, and `npm run type-check` for normal code changes.
- Use `npm run test:unit` or `npm run test:unit:coverage` for logic and component behavior covered by Vitest.
- Use `npm run test:component` or `npm run test:e2e` only when the task reaches Playwright coverage.
- Use `npm run build` before finishing broader UI changes.

## API Type Notes

- `npm run build:api` reads `src/api/openapi-spec.json` and regenerates `src/api/openapi-spec.d.ts`.
- `npm run update-spec` downloads the spec from `http://localhost:7860` and then rebuilds the TypeScript types.
- Use `$otx-openapi-sync` when backend API changes are part of the task.

## Coordination Notes

- The `preinstall` script clones pinned Geti UI workspace packages into `packages/`; do not edit those generated packages unless the task explicitly targets them.
- Keep generated API artifacts and the consuming UI changes in the same change set when the contract changes.
