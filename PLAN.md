# Build plan

Spec for building the site on top of this scaffold. Wireframes: https://claude.ai/artifact/VWvgRjjH4aBbNZEyG3zvkt

## Goal

A minimal personal site for fellowship applications and similar: who I am, what I've built, what I've written. Function over looks.

## Decisions

- **Stack:** Astro 7, static output, deployed to GitHub Pages by `.github/workflows/deploy.yml`.
- **URL:** https://vatslgoswami.github.io (repo `vatslgoswami/vatslgoswami.github.io`).
- **Pages:** About (`/`), Projects (`/projects`), Blogs (`/blog`), Shelf (`/shelf`), plus `/blog/<slug>` for posts written on the site.
- **Style:** minimal and text-first, like ishanjmukherjee.github.io. White background, near-black text. Mustard (`--mustard`) and salmon (`--salmon`) are accents only: links, tags, the active nav item, year labels. Tokens are in `src/styles/global.css`. IBM Plex Sans for text and IBM Plex Mono for dates and meta, via Google Fonts with system fallbacks. Light theme only.
- **No extras for now:** no publications, experience, news or tag filters.

UI implementation conventions and the typography scale are documented in [DESIGN.md](DESIGN.md).

## Shared layout

- `src/layouts/Base.astro`: html shell, `<title>`, meta description, favicon, Google Fonts link, global.css.
- Header: name on the left (links to `/`), nav on the right (`site.nav`). The active item gets a mustard underline.
- Content column uses `--measure`, centred, with a 16px minimum side gutter. Long-form text uses `--reading-measure`. Must work at phone width.
- Footer: `© <year> Vatsl Krishnn Goswami`.

## About (`src/pages/index.astro`)

- Two columns: square photo (`site.photo`, ~140px) on the left, bio on the right. Stacks on phones.
- Bio comes from `src/data/about.md` (rendered Markdown).
- Under the bio, a row of text chips for each `site.socials` entry, plus a highlighted mustard "CV (PDF)" chip linking to `site.cv`.
- Below, in this order: "Recent writing" (3 newest posts, same date + title rows as the Blogs list), then "Selected projects" (2 featured projects, newest first). Projects use the full Projects-page row component (meta line, title, summary, tags, links, thumbnail), so build that row as a shared component. Each section has an "all →" link to its page.
- If `public/photo.jpg` is missing, show a neutral placeholder rather than a broken image.

## Projects (`src/pages/projects.astro`)

- Medium-style rows, newest first (by `year`). Left: meta line `year · kind` in mono, bold title, summary clamped to 2 lines, then mustard tag pills and salmon link labels (`links`). Right: 3:2 thumbnail from `image`, if one is set.
- A project with `detail: true` links its title to its detail page; otherwise the title links to its first entry in `links`, if present. GitHub links appear beside the title.

## Blogs (`src/pages/blog/index.astro`, `src/pages/blog/[slug].astro`)

- Plain list, newest first: small grey year and Blog label above the title, matching project rows.
- Posts with `externalUrl` link out and show a small "↗" marker. The rest link to `/blog/<id>`.
- Post page: title, date, Markdown body in a ~65-character column, "← All posts" link. Generate pages only for posts without `externalUrl`.

## Shelf (`src/pages/shelf.astro`)

- A reading log of papers and posts, with a few lines on why I like each one. This is option B in the wireframes.
- An intro line (`site.shelfIntro`), followed by items grouped by the year they were added. Each item links its title, names its author when available, and displays its notes as nested arrows.
- One Markdown file per item in `src/content/shelf/`.

## Content files

- Files starting with `_` in `src/content/*/` are skipped by the site. Each folder has a `_template.md` to copy.

## Done when

- `npm run build` passes and all four routes render with the sample content.
- The layout works at 375px wide with no horizontal scroll.
- `draft: true` entries are hidden everywhere.
- The deploy workflow publishes the site after a push to `main`. In the repo, Settings → Pages → Source must be set to "GitHub Actions".

## How to work

Follow the git workflow in `AGENTS.md`: work on a branch (e.g. `feat/site-layout`), open a PR into `main`, wait for the "Build check" CI job to pass, then squash-merge. Never push to `main` directly, because every push to `main` deploys the live site.
