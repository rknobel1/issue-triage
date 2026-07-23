from evaluation.build_corpus import sampled_issue_numbers


def test_issue_number_sample_is_deterministic_and_bounded():
    first = sampled_issue_numbers(max_issue_number=100, attempt_limit=20, seed=42)
    second = sampled_issue_numbers(max_issue_number=100, attempt_limit=20, seed=42)

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    assert all(1 <= number <= 100 for number in first)


def test_issue_number_sample_does_not_exceed_history():
    sampled = sampled_issue_numbers(max_issue_number=3, attempt_limit=10, seed=42)

    assert sorted(sampled) == [1, 2, 3]
