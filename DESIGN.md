# Website UI conventions

Keep the site minimal, readable, and consistent. Use IBM Plex Sans for body text and headings, and IBM Plex Mono for dates and metadata. White and near-black form the base; mustard and salmon are accents. Define shared values in `src/styles/global.css` and use the existing tokens rather than introducing one-off sizes or colors.

## Typography

| Role | Token | Size |
| --- | --- | --- |
| Page title | `--text-page-title` | 28px |
| Year group heading | `--text-year` | 22px |
| Project and blog list title | `--text-row-title` | 19px |
| Body and shelf notes | `--text-body` | 17px |
| Secondary copy and back links | `--text-secondary` | 15px |
| Detail-page metadata | `--text-meta` | 13px |
| Code blocks | `--text-code` | 14px |
| List metadata and chips | `--text-small` | 12px |
| Tags and compact labels | `--text-micro` | 11px |

Use `.year-label` and `.year-heading` for year group headings. Use `YearMeta.astro` for the common year-and-kind row; list metadata uses the small, grey `--text-small` style rather than the heading style. `ContentTitle.astro` provides the shared project and blog list title and project detail title. Use `.page-title` and `.page-meta` for other detail headings. `ArticleBody.astro` renders detail content and the back link with a shared reading width; its styles use `.prose` and `.back-link`.

## Components and implementation

- Reuse `ProjectRow.astro` for project listings and `PostRow.astro` for blog listings, including on the home page. Both list rows use `.list-row` for spacing and dividers. Keep shared navigation and page chrome in `Base.astro`.
- Extract a component when markup and behavior are repeated; use shared CSS tokens or utility classes for repeated visual rules. Avoid abstractions for a single use.
- Keep content in the relevant content collection, and keep semantic headings and links accessible. Do not duplicate content in page templates.
- Keep layouts usable at 375px without horizontal scrolling. Let long titles and links wrap naturally; test the home page, lists, and detail pages when changing shared styles.
- Run `npm run build` before pushing. Keep `PLAN.md` current if the intended site structure or visual direction changes.
