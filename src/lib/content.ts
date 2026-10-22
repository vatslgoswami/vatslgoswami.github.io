import { getCollection } from 'astro:content';

// Published posts, newest first.
export async function getPosts() {
	const posts = await getCollection('blog', ({ data }) => !data.draft);
	return posts.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

// Published projects, newest first (by year, then title).
export async function getProjects() {
	const projects = await getCollection('projects', ({ data }) => !data.draft);
	return projects.sort(
		(a, b) => b.data.year - a.data.year || a.data.title.localeCompare(b.data.title),
	);
}
