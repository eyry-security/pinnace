"""Host-binding guard: exact host, wildcard subdomains, reference lookups."""

from __future__ import annotations

from pinnace.target_guard import TargetBinding, extract_hosts


def test_extract_hosts_ignores_files_and_attributes():
    hosts = extract_hosts(
        "curl https://a.example/x result.get cookies.txt scratch/out.json")
    assert hosts == ["a.example"]


def test_exact_host_binding_rejects_another_host():
    binding = TargetBinding.from_host("a.example")
    assert binding.check_call("shell", {"command": "curl https://a.example"}) is None
    reason = binding.check_call(
        "shell", {"command": "curl https://b.example"})
    assert reason and "out of scope" in reason
    assert "a.example" in reason and "b.example" in reason


def test_wildcard_binding_allows_only_its_subdomains():
    binding = TargetBinding.from_host("*.example.test")
    assert binding.check_call("nuclei", {"target": "https://a.example.test"}) is None
    assert binding.check_call("nuclei", {"target": "https://b.example.test"}) is None
    assert binding.check_call("nuclei", {"target": "https://other.test"})


def test_reference_lookup_is_allowed_but_not_as_a_scan_target():
    binding = TargetBinding.from_host("a.example")
    assert binding.check_call(
        "fetch_url", {"url": "https://crt.sh/?q=a.example"}) is None
    assert binding.check_call(
        "shell", {"command": "curl https://crt.sh/?q=a.example"}) is None
    assert binding.check_call(
        "nuclei", {"target": "https://crt.sh"}) is not None


def test_delegate_brief_cannot_name_another_host():
    binding = TargetBinding.from_host("a.example")
    reason = binding.check_call(
        "delegate_task",
        {"task": "test the lab hosts", "context": "b.example"},
    )
    assert reason and "b.example" in reason
    assert binding.check_call(
        "delegate_task",
        {"task": "check headers", "context": "reference: https://crt.sh"},
    ) is None


def test_tool_output_filters_other_hosts_but_keeps_reference_lines():
    binding = TargetBinding.from_host("a.example")
    text = (
        "GET https://a.example/ 200\n"
        "GET https://b.example/ 200\n"
        "crt.sh: certificate for a.example\n"
    )
    filtered = binding.filter_output(text)
    assert "a.example" in filtered
    assert "b.example" not in filtered
    assert "crt.sh" in filtered
    assert "filtered 1 out-of-scope" in filtered


def test_finish_payload_is_not_target_validated():
    binding = TargetBinding.from_host("a.example")
    assert binding.check_call(
        "finish", {"result": '{"findings": ["b.example"]}'}) is None
