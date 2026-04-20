from __future__ import annotations

import asyncio

from rag_core.demo import DemoHit, run_demo_app


async def run_demo() -> None:
    result = await run_demo_app()
    print(f"Indexed document: {result['document_id']} ({result['chunk_count']} chunks)")
    print("Top hits:")
    for hit in result["hits"]:
        typed_hit: DemoHit = hit
        print(f"- {typed_hit['score']:.3f} {typed_hit['title']}: {typed_hit['text'][:80]}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
