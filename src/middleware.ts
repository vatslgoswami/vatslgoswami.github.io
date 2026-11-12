import { defineMiddleware } from 'astro:middleware';
import { parse, serialize, type DefaultTreeAdapterTypes } from 'parse5';

// Apply the same link behavior to Astro components and rendered Markdown.
export const onRequest = defineMiddleware(async (context, next) => {
	const response = await next();
	if (!response.headers.get('content-type')?.includes('text/html') || !response.body) return response;

	const document = parse(await response.text());
	const localOrigins = new Set([context.url.origin, context.site?.origin].filter(Boolean));

	function visit(node: DefaultTreeAdapterTypes.Node) {
		if ('tagName' in node && node.tagName === 'a') {
			const href = node.attrs.find((attr) => attr.name === 'href')?.value;
			if (href) {
				try {
					const url = new URL(href, context.url);
					const external = ['http:', 'https:'].includes(url.protocol) && !localOrigins.has(url.origin);
					node.attrs = node.attrs.filter((attr) => attr.name !== 'target');
					if (external) {
						node.attrs.push({ name: 'target', value: '_blank' });
						const rel = node.attrs.find((attr) => attr.name === 'rel');
						if (rel) rel.value = [...new Set([...rel.value.split(/\s+/).filter(Boolean), 'noopener', 'noreferrer'])].join(' ');
						else node.attrs.push({ name: 'rel', value: 'noopener noreferrer' });
					}
				} catch {
					// Leave invalid URLs unchanged.
				}
			}
		}
		if ('childNodes' in node) node.childNodes.forEach(visit);
		if ('content' in node && node.content) visit(node.content);
	}
	visit(document);
	const headers = new Headers(response.headers);
	headers.delete('content-length');
	return new Response(serialize(document), { status: response.status, statusText: response.statusText, headers });
});
