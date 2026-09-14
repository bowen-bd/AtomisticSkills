import importlib.util
from pathlib import Path

# Load check_publisher from hyphenated directory skills/general-publisher-access-guard
script_path = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "general-publisher-access-guard"
    / "scripts"
    / "check_publisher.py"
)
spec = importlib.util.spec_from_file_location("check_publisher", script_path)
check_publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_publisher)

load_policy_matrix = check_publisher.load_policy_matrix
identify_publisher = check_publisher.identify_publisher
extract_doi_prefix = check_publisher.extract_doi_prefix

from src.utils.paper_downloader import _is_url_blocked


def test_publisher_policy_matrix_structure():
    matrix = load_policy_matrix()
    assert "publishers" in matrix
    assert len(matrix["publishers"]) >= 20

    for pub in matrix["publishers"]:
        assert "id" in pub
        assert "name" in pub
        assert "domains" in pub
        assert len(pub["domains"]) > 0
        assert "direct_scrape_policy" in pub
        assert pub["direct_scrape_policy"] in (
            "blocked",
            "allowed_api_only",
            "throttled",
            "allowed",
        )
        assert "waf_type" in pub
        assert "recommended_api" in pub


def test_identify_blocked_publishers_by_url():
    matrix = load_policy_matrix()

    test_cases = [
        ("https://pubs.acs.org/doi/10.1021/jacs.3c01234", "acs", "blocked"),
        (
            "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.010401",
            "aps",
            "blocked",
        ),
        ("https://link.aps.org/pdf/10.1103/PhysRevLett.120.145301", "aps", "blocked"),
        (
            "https://www.sciencedirect.com/science/article/pii/S092583882300001X",
            "elsevier",
            "blocked",
        ),
        (
            "https://www.nature.com/articles/s41586-023-00001-w",
            "springer_nature",
            "blocked",
        ),
        (
            "https://link.springer.com/article/10.1007/s10853-023-00001-1",
            "springer_nature",
            "blocked",
        ),
        (
            "https://onlinelibrary.wiley.com/doi/10.1002/anie.202300001",
            "wiley",
            "blocked",
        ),
        (
            "https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc00001a",
            "rsc",
            "blocked",
        ),
        (
            "https://iopscience.iop.org/article/10.1088/1361-648X/ac1234",
            "iop",
            "blocked",
        ),
        ("https://pubs.aip.org/jcp/article/158/1/010901/2869000", "aip", "blocked"),
        ("https://ieeexplore.ieee.org/document/10000001", "ieee", "blocked"),
        (
            "https://chemrxiv.org/engage/chemrxiv/article-details/60c74b0b14643f80c6bebfd5",
            "chemrxiv",
            "allowed_api_only",
        ),
        ("https://arxiv.org/abs/2301.00001", "arxiv", "allowed_api_only"),
    ]

    for url, expected_id, expected_policy in test_cases:
        pub = identify_publisher(url, matrix)
        assert pub is not None, f"Failed to match publisher for {url}"
        assert (
            pub["id"] == expected_id
        ), f"Expected {expected_id} for {url}, got {pub['id']}"
        assert pub["direct_scrape_policy"] == expected_policy


def test_identify_publishers_by_doi():
    matrix = load_policy_matrix()

    test_cases = [
        ("10.1021/acs.chemmater.9b04758", "acs"),
        ("10.1103/PhysRevLett.120.145301", "aps"),
        ("10.1016/j.commatsci.2021.110520", "elsevier"),
        ("10.1038/nature14539", "springer_nature"),
        ("10.1002/anie.202300001", "wiley"),
        ("10.1039/D0TA07271B", "rsc"),
        ("10.1088/1361-648X/aa5e58", "iop"),
        ("10.1109/TPAMI.2020.1234567", "ieee"),
    ]

    for doi, expected_id in test_cases:
        pub = identify_publisher(doi, matrix)
        assert pub is not None, f"Failed to match publisher for DOI {doi}"
        assert pub["id"] == expected_id


def test_is_url_blocked_filtering():
    assert (
        _is_url_blocked("https://link.aps.org/pdf/10.1103/PhysRevLett.120.145301")
        is True
    )
    assert _is_url_blocked("https://pubs.acs.org/doi/pdf/10.1021/jacs.3c01234") is True
    assert (
        _is_url_blocked(
            "https://www.sciencedirect.com/science/article/pii/S092702562100520X"
        )
        is True
    )
    assert (
        _is_url_blocked(
            "https://onlinelibrary.wiley.com/doi/pdf/10.1002/anie.202300001"
        )
        is True
    )

    # Safe repositories must NOT be blocked
    assert _is_url_blocked("https://arxiv.org/pdf/2301.00001.pdf") is False
    assert (
        _is_url_blocked(
            "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4336040&blobtype=pdf"
        )
        is False
    )
    assert (
        _is_url_blocked(
            "https://dspace.mit.edu/bitstreams/b6249448-f14f-4805-bf14-3e013cd01173/download"
        )
        is False
    )
    assert (
        _is_url_blocked("https://zenodo.org/record/123456/files/article.pdf") is False
    )
    assert _is_url_blocked("https://www.osti.gov/servlets/purl/1524040") is False
