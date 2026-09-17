import numpy as np
import pytest

from messina_info.embeddings import (
    normalize_vectors,
    prepare_e5_passages,
    prepare_e5_queries,
)


def test_prepares_e5_queries_and_passages() -> None:
    assert prepare_e5_queries(["  student deadline "]) == [
        "query: student deadline"
    ]
    assert prepare_e5_passages([" Scholarship details "]) == [
        "passage: Scholarship details"
    ]


@pytest.mark.parametrize("prepare", [prepare_e5_queries, prepare_e5_passages])
def test_e5_preparation_rejects_empty_text(prepare: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare(["  "])  # type: ignore[operator]


def test_normalizes_vectors_as_float32() -> None:
    vectors = normalize_vectors(np.array([[3.0, 4.0], [0.0, 2.0]]))

    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_rejects_zero_or_wrong_sized_vectors() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        normalize_vectors(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="expected 2"):
        normalize_vectors(np.array([[1.0, 0.0]]), expected_rows=2)
