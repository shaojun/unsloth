# AGENTS.md

## i18n (studio/frontend)

- `src/i18n/locales/en.ts` is the source of truth; keys are typed, so `tsc` fails on any `t("…")` key missing from `en`.
- **Team decision: only English and Chinese (zh-CN) get real translations.** Other locales (ja, ko, de, fr, es, it, pt-br, ru, ar, hi) are maintained as English-fallback copies — do not spend effort translating them.
- CI runs `npm run i18n:check:strict`, which fails if *any* en key is missing from *any* locale file, if placeholders (`{name}`, `{count}`) mismatch, or if a locale has keys en doesn't. Workflow when adding UI strings: add keys to `en.ts` → translate in `zh-CN.ts` → copy the en strings into the other 10 locale files (same block, same key set).
