# vatslgoswami.github.io

Personal site built with [Astro](https://astro.build), deployed to GitHub Pages.

## Run locally

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # outputs to dist/
```

## Editing content

| What | Where |
|---|---|
| Name, email, social links, nav | `src/data/site.ts` |
| Bio on the About page | `src/data/about.md` |
| Photo | `public/photo.jpg` |
| CV | `public/cv.pdf` |
| Projects | one `.md` file per project in `src/content/projects/` |
| Blog posts | one `.md` file per post in `src/content/blog/` |
| Shelf (papers and posts you've read) | one `.md` file per item in `src/content/shelf/`; notes go in the body |
| Project images | `public/images/projects/` |

Each content folder has a `_template.md` to copy; files starting with `_` are ignored. Field definitions live in `src/content.config.ts`. A blog post with `externalUrl` (e.g. a Medium link) is listed but not hosted here. Set `draft: true` to hide any entry.

## Deploying

Pushing to `main` runs `.github/workflows/deploy.yml`. One-time setup: in the GitHub repo, set Settings → Pages → Source to **GitHub Actions**.

See `PLAN.md` for the design spec.
