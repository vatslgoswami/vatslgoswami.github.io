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

## Git workflow (required)

Never commit or push directly to `main`. Every push to `main` deploys the live site.

1. Start from an up-to-date `main`: `git checkout main && git pull`.
2. Create a branch for the task: `feat/<short-name>`, `fix/<short-name>` or `chore/<short-name>`.
3. Commit in small, focused steps with clear messages.
4. Run `npm run build` and make sure it passes before pushing.
5. Push the branch and open a pull request into `main` with `gh pr create`. In the description, say what changed and how you checked it, and list anything left undone.
6. Wait for the "Build check" CI job to pass. Fix failures on the same branch.
7. Merge with `gh pr merge --squash --delete-branch` only after CI is green. If the task brief says the owner will review first, leave the PR open instead.
8. After merging, confirm the "Deploy to GitHub Pages" run on `main` succeeds.
