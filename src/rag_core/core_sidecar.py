from __future__ import annotations

from rag_core.search.result_payload import payload_to_result
from rag_core.search.lexical_sidecar import LexicalSidecarRecord


def build_sidecar_records(
    *,
    namespace: str,
    point_ids: list[str],
    point_payloads: list[dict[str, object]],
) -> list[LexicalSidecarRecord]:
    return [
        LexicalSidecarRecord(
            namespace=namespace,
            result=payload_to_result(
                point_id=point_id,
                payload=payload,
                score=0.0,
            ),
        )
        for point_id, payload in zip(point_ids, point_payloads, strict=True)
    ]
