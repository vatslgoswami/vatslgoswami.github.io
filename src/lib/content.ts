import { getCollection } from 'astro:content';

// Published posts, newest first.
export async function getPosts() {
	const posts = await getCollection('blog', ({ data }) => !data.draft);
	return posts.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

// Published projects, pinned first, then newest first (by year, then title).
export async function getProjects() {
	const projects = await getCollection('projects', ({ data }) => !data.draft);
	return projects.sort(
		(a, b) => Number(b.data.pinned) - Number(a.data.pinned) || b.data.year - a.data.year || a.data.title.localeCompare(b.data.title),
	);
}

// Published projects visible in site listings; unlisted detail pages remain accessible.
export async function getListedProjects() {
	return (await getProjects()).filter((project) => project.data.listed !== false && !project.data.research);
}

// Research write-ups can come from either collection while sharing one listing.
export async function getResearchEntries() {
	const [posts, projects] = await Promise.all([getPosts(), getProjects()]);
	const entries = [
		...posts.map((entry) => ({ type: 'blog' as const, entry, sortDate: entry.data.date.getTime() })),
		...projects.filter((entry) => entry.data.research && entry.data.detail)
			.map((entry) => ({ type: 'project' as const, entry, sortDate: Date.UTC(entry.data.year, 0, 1) })),
	];
	const slugs = new Set<string>();
	for (const item of entries) {
		if (slugs.has(item.entry.id)) throw new Error(`Duplicate research slug: ${item.entry.id}`);
		slugs.add(item.entry.id);
	}
	return entries.sort((a, b) => b.sortDate - a.sortDate);
}

// Published shelf items, most recently added first.
export async function getShelf() {
	const items = await getCollection('shelf', ({ data }) => !data.draft);
	return items.sort((a, b) => b.data.added.getTime() - a.data.added.getTime() || a.id.localeCompare(b.id));
}
