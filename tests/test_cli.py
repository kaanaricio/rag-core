import json

import pytest

from rag_core.cli import main


def test_doctor_json_reports_planned_runtime(capsys) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_name"] == "rag_core_chunks__text_embedding_3_small_1536d"
    assert payload["embedding"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
    assert payload["reranker"]["effective"] == "none"
    assert "pdf_inspector" in payload


def test_doctor_check_store_creates_local_collection(capsys) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--check-store",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["store_health"]["healthy"] is True
    assert payload["store_health"]["collection"] == "rag_core_chunks__text_embedding_3_small_1536d"


def test_demo_json_runs_without_external_services(capsys) -> None:
    exit_code = main(["demo", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_count"] > 0
    assert payload["hits"]


def test_manifest_json_previews_file(tmp_path, capsys) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    exit_code = main(
        [
            "manifest",
            str(file_path),
            "--namespace",
            "acme",
            "--corpus-id",
            "help-center",
            "--metadata",
            "source=seed",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_entry"]["filename"] == "guide.txt"
    assert payload["manifest_entry"]["parser"] == "local:text"
    assert payload["manifest_entry"]["metadata"]["source"] == "seed"
    assert payload["document"]["ingest_state"] == "preview"


def test_doctor_default_output_is_human_readable(capsys) -> None:
    exit_code = main(
        [
            "doctor",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Collection:" in output
    assert "Embedding:" in output
    assert not output.lstrip().startswith("{")


def test_manifest_missing_file_reports_cli_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "manifest",
                "does-not-exist.txt",
                "--namespace",
                "acme",
                "--corpus-id",
                "help-center",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "file not found" in error
    assert "Traceback" not in error


def test_manifest_bad_metadata_reports_cli_error(tmp_path, capsys) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "manifest",
                str(file_path),
                "--namespace",
                "acme",
                "--corpus-id",
                "help-center",
                "--metadata",
                "broken",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "metadata entries must use KEY=VALUE" in error
    assert "Traceback" not in error
