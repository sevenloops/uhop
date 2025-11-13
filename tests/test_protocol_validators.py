from uhop.protocol import PROTOCOL_VERSION, ValidationResult, validate_incoming


def test_validator_rejects_missing_type():
    msg = {"v": PROTOCOL_VERSION}
    res = validate_incoming(msg)
    assert isinstance(res, ValidationResult)
    assert not res.ok
    assert "type" in (res.reason or "").lower()


def test_validator_rejects_unknown_action():
    msg = {"v": PROTOCOL_VERSION, "type": "request", "id": "1", "action": "totally_unknown", "params": {}}
    res = validate_incoming(msg)
    assert not res.ok
    assert "unknown action" in (res.reason or "").lower()


def test_validator_accepts_hello():
    msg = {"v": PROTOCOL_VERSION, "type": "hello", "agent": "test", "version": "0.0"}
    res = validate_incoming(msg)
    assert res.ok


def test_validator_response_ok_needs_data():
    msg = {"v": PROTOCOL_VERSION, "type": "response", "id": "x", "ok": True}
    res = validate_incoming(msg)
    assert not res.ok
    assert "data" in (res.reason or "").lower()
