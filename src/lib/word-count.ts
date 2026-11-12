import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import { parseFragment, type DefaultTreeAdapterTypes } from 'parse5';

// Count prose, headings, inline code, and figure captions; omit markup, URLs,
// code blocks, image descriptions, media fallback text, and LaTeX source.
export function countArticleWords(markdown: string): number {
	const fragments: string[] = [];
	const tree = unified().use(remarkParse).use(remarkMath).parse(markdown);

	function captionText(node: DefaultTreeAdapterTypes.Node, insideCaption = false): void {
		const inCaption = insideCaption || ('tagName' in node && node.tagName === 'figcaption');
		if (inCaption && node.nodeName === '#text' && 'value' in node) fragments.push(node.value);
		if ('childNodes' in node) node.childNodes.forEach((child) => captionText(child, inCaption));
	}

	function visit(node: unknown): void {
		if (!node || typeof node !== 'object' || !('type' in node)) return;
		const current = node as { type: string; value?: unknown; children?: unknown[] };
		if ((current.type === 'text' || current.type === 'inlineCode') && typeof current.value === 'string') fragments.push(current.value);
		else if (current.type === 'html' && typeof current.value === 'string') captionText(parseFragment(current.value));
		else if (current.type !== 'code' && current.type !== 'math' && current.type !== 'inlineMath') current.children?.forEach(visit);
	}
	visit(tree);
	return (fragments.join(' ').match(/[\p{L}\p{N}]+(?:[’'\-][\p{L}\p{N}]+)*/gu) ?? []).length;
}
