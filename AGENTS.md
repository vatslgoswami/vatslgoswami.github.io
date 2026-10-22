## Development

When starting the dev server, use background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)

## This project

Personal site for Vatsl Krishnn Goswami, deployed to https://vatslgoswami.github.io. `PLAN.md` is the source of truth for pages, layout and style; read it before changing anything visual.

- Content schema: `src/content.config.ts` (projects, blog). Import `z` from `astro/zod`, not `astro:content` (deprecated in Astro 7).
- Site-wide data: `src/data/site.ts`; bio: `src/data/about.md`.
- Style tokens: `src/styles/global.css`. Keep the look minimal; mustard and salmon are accents only.
- Verify with `npm run build` before committing.
