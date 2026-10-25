import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// One Markdown file per project in src/content/projects/ (files starting with _ are skipped)
const projects = defineCollection({
	loader: glob({ pattern: '**/[^_]*.md', base: './src/content/projects' }),
	schema: z.object({
		title: z.string(),
		year: z.number(),
		// e.g. "Research", "Internship", "Side project", "Course project"
		kind: z.string(),
		summary: z.string(),
		tags: z.array(z.string()).default([]),
		// Path under public/, e.g. "/images/projects/my-project.png"
		image: z.string().optional(),
		links: z
			.array(z.object({ label: z.string(), url: z.string().url() }))
			.default([]),
		featured: z.boolean().default(false),
		draft: z.boolean().default(false),
	}),
});

// One Markdown file per post in src/content/blog/ (files starting with _ are skipped).
// Set externalUrl to list a post hosted elsewhere (e.g. Medium); the body can then be empty.
const blog = defineCollection({
	loader: glob({ pattern: '**/[^_]*.md', base: './src/content/blog' }),
	schema: z.object({
		title: z.string(),
		date: z.coerce.date(),
		description: z.string().optional(),
		externalUrl: z.string().url().optional(),
		draft: z.boolean().default(false),
	}),
});

// One Markdown file per paper/post in src/content/shelf/ (files starting with _ are skipped).
// The body holds your notes on why you like it; Markdown and links allowed.
const shelf = defineCollection({
	loader: glob({ pattern: '**/[^_]*.md', base: './src/content/shelf' }),
	schema: z.object({
		title: z.string(),
		url: z.string().url(),
		author: z.string().optional(),
		// e.g. "paper", "blog", "book", "talk"
		kind: z.string(),
		// Year it was published (optional)
		year: z.number().optional(),
		// When you added it to the shelf; sets the order and the year groups
		added: z.coerce.date(),
		draft: z.boolean().default(false),
	}),
});

export const collections = { projects, blog, shelf };
