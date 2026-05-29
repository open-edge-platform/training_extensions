---
applyTo: "application/ui/**"
---

## Conventions

- Source in `application/ui/src/`. Feature folders group component + hooks + tests.
- TypeScript: no `any`. Use `unknown` + narrow. `type` for unions, `interface`
  for extensible object shapes.
- Function components + hooks only. Co-locate styles.
- Server state via React Query (`useQuery` / `useMutation`) — never call
  `fetch` / `axios` directly from components.
- **API types are generated** (`src/api/openapi-spec.json` / `.d.ts`) — never
  edit by hand. Regenerate: `npm run update-spec` (backend on `:7860`) or
  `npm run build:api` (if you already have a fresh local `openapi-spec.json`).
- **Vendored packages** (`packages/config`, `packages/ui`, `packages/smart-tools`)
  are cloned via `npm run clone-geti-ui-packages`. Do not edit locally.

## Commands

| Task                | Command                  |
| ------------------- | ------------------------ |
| Dev server          | `npm run start`          |
| Production build    | `npm run build`          |
| Desktop dev (Tauri) | `npm run start:desktop`  |
| Lint                | `npm run lint`           |
| Lint + fix          | `npm run lint:fix`       |
| Type-check          | `npm run type-check`     |
| Unit tests          | `npm run test:unit`      |
| Component tests     | `npm run test:component` |
| E2E tests           | `npm run test:e2e`       |
| Update API types    | `npm run update-spec`    |

## Testing

- Unit tests next to source as `*.test.ts(x)` (Vitest).
- Component / E2E under `tests/` (Playwright).
- Mocks in `mocks/` — reuse existing handlers. New API endpoints need a
  corresponding mock handler.

## Styling

- Use CSS Modules (`.module.scss`) co-located with components.

## Do not

- Do not introduce another data-fetching library.
- Do not hand-edit generated API types.
- Do not edit vendored `packages` — they will be overwritten.
- Do not add CSS-in-JS outside what `@geti/ui` already uses.

## Further reading

- See `application/ui/README.md` for detailed architecture, API integration examples, and contributing guidelines.
