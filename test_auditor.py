from auditor import Auditor

def test_mock_auditor():
    auditor = Auditor(mock_mode=True)

    print("Testing Mock Auditor...")

    # Test 1: Pass
    status, msg = auditor.audit_request("Limit EMEA to 50", "passport...", [])
    print(f"1. Normal: {status} - {msg}")
    assert status == "PASS"

    # Test 2: Ambiguity
    status, msg = auditor.audit_request("Move the big ones", "passport...", [])
    print(f"2. Ambiguity: {status} - {msg}")
    assert status == "CLARIFICATION_NEEDED"

    # Test 3: Hallucination
    status, msg = auditor.audit_request("Select unicorns", "passport...", [])
    print(f"3. Hallucination: {status} - {msg}")
    assert status == "FAIL"

    # Test 4: Logic Conflict
    existing = [{"description": "Blackout Jan due to maintenance"}]
    status, msg = auditor.audit_request("Schedule in Jan", "passport...", existing)
    print(f"4. Logic: {status} - {msg}")
    assert status == "FAIL"

    print("All tests passed!")

if __name__ == "__main__":
    test_mock_auditor()
