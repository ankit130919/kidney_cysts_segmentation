"""The environment table. Picking the wrong archive is not a recoverable mistake."""
import pytest

from cysts.common import download_dicoms as dl


def test_all_four_environments_present_with_the_right_auth():
    assert set(dl.ENVIRONMENTS) == {"prod", "staging", "qa", "sandbox"}
    assert dl.ENVIRONMENTS["prod"]["auth"] is True
    for e in ("staging", "qa", "sandbox"):
        assert dl.ENVIRONMENTS[e]["auth"] is False, f"{e} must not require a token"


def test_urls_match_the_published_table():
    assert dl.ENVIRONMENTS["prod"]["url"] == "https://dcm.5cnetwork.com/download/"
    assert dl.ENVIRONMENTS["staging"]["url"] == \
        "https://e2e-staging-api.5cnetwork.com/dicom/download/"
    assert dl.ENVIRONMENTS["qa"]["url"] == "https://e2e-qa-api.5cnetwork.com/dicom/download/"
    assert dl.ENVIRONMENTS["sandbox"]["url"] == \
        "https://e2e-sandbox-api.5cnetwork.com/dicom/download/"


def test_unknown_env_is_refused_and_does_not_fall_back_to_prod():
    with pytest.raises(ValueError) as e:
        dl.resolve("stagin")
    assert "unknown env" in str(e.value)


def test_default_is_prod():
    assert dl.DEFAULT_ENV == "prod"


def test_open_environments_resolve_without_a_token(monkeypatch):
    monkeypatch.delenv("NCCTF_DICOM_TOKEN", raising=False)
    monkeypatch.delenv("CYSTS_DICOM_TOKEN", raising=False)
    for e in ("staging", "qa", "sandbox"):
        assert dl.resolve(e)["env"] == e


def test_prod_without_a_token_fails_loudly_rather_than_sending_an_unauthed_request(monkeypatch):
    monkeypatch.delenv("NCCTF_DICOM_TOKEN", raising=False)
    monkeypatch.delenv("CYSTS_DICOM_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as e:
        dl.resolve("prod")
    assert "NCCTF_DICOM_TOKEN" in str(e.value)
