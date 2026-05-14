import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.src.core.tools.phases.research_tools.tools import (
    _format_pubmed_article,
    _format_trial,
    _get_with_retry,
    _parse_pubmed_article,
    get_pubmed_abstract,
    get_trial_details,
    search_clinical_trials,
    search_pubmed,
)


class MockRuntime:
    state = {"agent_id": "test_agent", "disease_name": "Alzheimer"}
    tool_call_id = "test_call"


def _mock_response(data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data) if isinstance(data, (dict, list)) else data
    mock.raise_for_status = MagicMock()
    return mock


def _passthrough_send_results(result, _runtime):
    return result


# ---------------------------------------------------------------------------
# _format_trial
# ---------------------------------------------------------------------------


class TestFormatTrial:
    def test_full_data(self):
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Alzheimer Drug Trial",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE2"]},
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG", "name": "Drug X"}]
                },
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Cognitive score at 12 months"}]
                },
                "contactsLocationsModule": {
                    "locations": [
                        {
                            "facility": "MGH",
                            "city": "Boston",
                            "state": "MA",
                            "country": "US",
                        }
                    ]
                },
            }
        }
        result = _format_trial(study)
        assert "NCT01234567" in result
        assert "Alzheimer Drug Trial" in result
        assert "RECRUITING" in result
        assert "PHASE2" in result
        assert "Drug X" in result
        assert "Cognitive score at 12 months" in result
        assert "MGH" in result
        assert "Boston" in result

    def test_missing_fields_returns_na(self):
        result = _format_trial({"protocolSection": {}})
        assert result.count("N/A") >= 4

    def test_empty_study(self):
        result = _format_trial({})
        assert "N/A" in result

    def test_multiple_phases(self):
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT999", "briefTitle": "T"},
                "statusModule": {},
                "designModule": {"phases": ["PHASE3", "PHASE4"]},
                "armsInterventionsModule": {},
                "outcomesModule": {},
                "contactsLocationsModule": {},
            }
        }
        result = _format_trial(study)
        assert "PHASE3, PHASE4" in result

    def test_locations_capped_at_three(self):
        locs = [
            {"facility": f"Site{i}", "city": "City", "state": "ST", "country": "US"}
            for i in range(5)
        ]
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1", "briefTitle": "T"},
                "statusModule": {},
                "designModule": {},
                "armsInterventionsModule": {},
                "outcomesModule": {},
                "contactsLocationsModule": {"locations": locs},
            }
        }
        result = _format_trial(study)
        assert result.count("Site") == 3

    def test_multiple_interventions_joined(self):
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1", "briefTitle": "T"},
                "statusModule": {},
                "designModule": {},
                "armsInterventionsModule": {
                    "interventions": [
                        {"type": "DRUG", "name": "Alpha"},
                        {"type": "DRUG", "name": "Beta"},
                    ]
                },
                "outcomesModule": {},
                "contactsLocationsModule": {},
            }
        }
        result = _format_trial(study)
        assert "Alpha, Beta" in result


# ---------------------------------------------------------------------------
# _parse_pubmed_article
# ---------------------------------------------------------------------------


class TestParsePubmedArticle:
    def test_full_article(self):
        xml_str = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Test Paper Title</ArticleTitle>
                    <Journal>
                        <Title>Nature Medicine</Title>
                        <JournalIssue>
                            <PubDate><Year>2024</Year></PubDate>
                        </JournalIssue>
                    </Journal>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <Initials>J</Initials>
                        </Author>
                        <Author>
                            <LastName>Doe</LastName>
                            <Initials>A</Initials>
                        </Author>
                    </AuthorList>
                    <Abstract>
                        <AbstractText>This is the abstract text.</AbstractText>
                    </Abstract>
                </Article>
                <MeshHeadingList>
                    <MeshHeading>
                        <DescriptorName>Alzheimer Disease</DescriptorName>
                    </MeshHeading>
                    <MeshHeading>
                        <DescriptorName>Amyloid</DescriptorName>
                    </MeshHeading>
                </MeshHeadingList>
            </MedlineCitation>
        </PubmedArticle>
        """
        elem = ET.fromstring(xml_str)
        result = _parse_pubmed_article(elem)
        assert result["pmid"] == "12345678"
        assert result["title"] == "Test Paper Title"
        assert "Smith J" in result["authors"]
        assert "Doe A" in result["authors"]
        assert result["journal"] == "Nature Medicine"
        assert result["year"] == "2024"
        assert "abstract text" in result["abstract"]
        assert "Alzheimer Disease" in result["mesh_terms"]
        assert "Amyloid" in result["mesh_terms"]

    def test_labeled_abstract_sections(self):
        xml_str = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>111</PMID>
                <Article>
                    <ArticleTitle>T</ArticleTitle>
                    <Journal><Title>J</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
                    <Abstract>
                        <AbstractText Label="BACKGROUND">BG text</AbstractText>
                        <AbstractText Label="CONCLUSION">Conc text</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        elem = ET.fromstring(xml_str)
        result = _parse_pubmed_article(elem)
        assert "BACKGROUND: BG text" in result["abstract"]
        assert "CONCLUSION: Conc text" in result["abstract"]

    def test_missing_citation_returns_empty(self):
        elem = ET.fromstring("<PubmedArticle></PubmedArticle>")
        assert _parse_pubmed_article(elem) == {}

    def test_missing_article_section(self):
        elem = ET.fromstring(
            "<PubmedArticle><MedlineCitation><PMID>999</PMID></MedlineCitation></PubmedArticle>"
        )
        result = _parse_pubmed_article(elem)
        assert result["pmid"] == "999"
        assert "title" not in result

    def test_medline_date_fallback(self):
        xml_str = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>222</PMID>
                <Article>
                    <ArticleTitle>T</ArticleTitle>
                    <Journal>
                        <Title>J</Title>
                        <JournalIssue>
                            <PubDate><MedlineDate>2021 Jan-Feb</MedlineDate></PubDate>
                        </JournalIssue>
                    </Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        elem = ET.fromstring(xml_str)
        result = _parse_pubmed_article(elem)
        assert result["year"] == "2021 Jan-Feb"


# ---------------------------------------------------------------------------
# _format_pubmed_article
# ---------------------------------------------------------------------------


class TestFormatPubmedArticle:
    def test_full_data(self):
        data = {
            "pmid": "12345678",
            "title": "Test Paper",
            "authors": ["Smith J", "Doe A"],
            "journal": "Nature Medicine",
            "year": "2024",
            "abstract": "Abstract text here.",
            "mesh_terms": ["Alzheimer Disease"],
        }
        result = _format_pubmed_article(data)
        assert "PMID: 12345678" in result
        assert "Test Paper" in result
        assert "Smith J" in result
        assert "Nature Medicine, 2024" in result
        assert "Abstract text here." in result
        assert "Alzheimer Disease" in result

    def test_author_truncation_with_et_al(self):
        data = {
            "pmid": "1",
            "title": "T",
            "authors": [f"Author{i}" for i in range(10)],
            "journal": "J",
            "year": "2020",
            "abstract": "A",
            "mesh_terms": [],
        }
        result = _format_pubmed_article(data)
        assert "et al." in result
        assert "Author6" not in result

    def test_missing_mesh_terms_omits_line(self):
        data = {
            "pmid": "2",
            "title": "T",
            "authors": [],
            "journal": "J",
            "year": "",
            "abstract": "A",
            "mesh_terms": [],
        }
        result = _format_pubmed_article(data)
        assert "MeSH" not in result

    def test_mesh_terms_capped_at_ten(self):
        data = {
            "pmid": "3",
            "title": "T",
            "authors": [],
            "journal": "J",
            "year": "",
            "abstract": "A",
            "mesh_terms": [f"Term{i}" for i in range(15)],
        }
        result = _format_pubmed_article(data)
        assert "Term10" not in result
        assert "Term9" in result


# ---------------------------------------------------------------------------
# _get_with_retry
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    def test_success_on_first_attempt(self):
        mock_resp = _mock_response({"ok": True})
        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = _get_with_retry("https://example.com", {"k": "v"})
            assert mock_get.call_count == 1
            assert result is mock_resp

    def test_retries_once_on_failure_then_succeeds(self):
        mock_resp = _mock_response({"ok": True})
        with patch(
            "requests.get",
            side_effect=[requests.RequestException("timeout"), mock_resp],
        ) as mock_get:
            result = _get_with_retry("https://example.com")
            assert mock_get.call_count == 2
            assert result is mock_resp

    def test_raises_after_two_failures(self):
        with patch(
            "requests.get", side_effect=requests.RequestException("fail")
        ):
            with pytest.raises(requests.RequestException):
                _get_with_retry("https://example.com")

    def test_passes_params_to_requests(self):
        mock_resp = _mock_response({})
        with patch("requests.get", return_value=mock_resp) as mock_get:
            _get_with_retry("https://example.com", {"key": "val"})
            _, kwargs = mock_get.call_args
            assert kwargs["params"] == {"key": "val"}
            assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# search_clinical_trials
# ---------------------------------------------------------------------------

_CT_PATCH = "app.src.core.tools.phases.research_tools.tools._get_with_retry"
_SR_PATCH = "app.src.core.tools.phases.research_tools.tools.send_results"


class TestSearchClinicalTrials:
    def test_query_params_construction(self):
        mock_resp = _mock_response({"studies": []})
        with patch(_CT_PATCH, return_value=mock_resp) as mock_get:
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                search_clinical_trials.func(
                    condition="Alzheimer",
                    status="RECRUITING",
                    phase="PHASE2",
                    max_results=5,
                    runtime=MockRuntime(),
                )
                _url, kwargs = mock_get.call_args
                params = kwargs["params"]
                assert params["query.cond"] == "Alzheimer"
                assert params["filter.overallStatus"] == "RECRUITING"
                assert params["filter.phase"] == "PHASE2"
                assert params["pageSize"] == 5

    def test_no_status_or_phase_omits_filters(self):
        mock_resp = _mock_response({"studies": []})
        with patch(_CT_PATCH, return_value=mock_resp) as mock_get:
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                search_clinical_trials.func(
                    condition="Alzheimer",
                    status=None,
                    phase=None,
                    max_results=10,
                    runtime=MockRuntime(),
                )
                _url, kwargs = mock_get.call_args
                params = kwargs["params"]
                assert "filter.overallStatus" not in params
                assert "filter.phase" not in params

    def test_empty_results_message(self):
        mock_resp = _mock_response({"studies": []})
        with patch(_CT_PATCH, return_value=mock_resp):
            result = search_clinical_trials.func(
                condition="ObscureDisease",
                status=None,
                phase=None,
                max_results=10,
                runtime=MockRuntime(),
            )
            assert "No clinical trials found" in result
            assert "ObscureDisease" in result

    def test_api_error_returns_message(self):
        with patch(_CT_PATCH, side_effect=requests.RequestException("503")):
            result = search_clinical_trials.func(
                condition="Alzheimer",
                status=None,
                phase=None,
                max_results=10,
                runtime=MockRuntime(),
            )
            assert "Error" in result

    def test_formats_trials_as_markdown(self):
        studies = [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT01234567",
                        "briefTitle": "Trial Title",
                    },
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "designModule": {"phases": ["PHASE2"]},
                    "armsInterventionsModule": {
                        "interventions": [{"type": "DRUG", "name": "Drug X"}]
                    },
                    "outcomesModule": {
                        "primaryOutcomes": [{"measure": "Cognitive score"}]
                    },
                    "contactsLocationsModule": {"locations": []},
                }
            }
        ]
        mock_resp = _mock_response({"studies": studies})
        with patch(_CT_PATCH, return_value=mock_resp):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = search_clinical_trials.func(
                    condition="Alzheimer",
                    status=None,
                    phase=None,
                    max_results=10,
                    runtime=MockRuntime(),
                )
                assert "## Clinical Trials" in result
                assert "NCT01234567" in result
                assert "RECRUITING" in result
                assert "PHASE2" in result
                assert "Drug X" in result
                assert "### 1." in result

    def test_multiple_trials_numbered(self):
        def make_study(nct_id):
            return {
                "protocolSection": {
                    "identificationModule": {"nctId": nct_id, "briefTitle": "T"},
                    "statusModule": {},
                    "designModule": {},
                    "armsInterventionsModule": {},
                    "outcomesModule": {},
                    "contactsLocationsModule": {},
                }
            }

        mock_resp = _mock_response(
            {"studies": [make_study("NCT001"), make_study("NCT002")]}
        )
        with patch(_CT_PATCH, return_value=mock_resp):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = search_clinical_trials.func(
                    condition="X",
                    status=None,
                    phase=None,
                    max_results=10,
                    runtime=MockRuntime(),
                )
                assert "### 1." in result
                assert "### 2." in result


# ---------------------------------------------------------------------------
# get_trial_details
# ---------------------------------------------------------------------------


class TestGetTrialDetails:
    def test_returns_full_detail_markdown(self):
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Full Trial",
                    "officialTitle": "Official Full Trial Name",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"phases": ["PHASE3"]},
                "descriptionModule": {"briefSummary": "This study tests X."},
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG", "name": "Compound A"}]
                },
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Survival at 6 months"}],
                    "secondaryOutcomes": [{"measure": "Quality of life"}],
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Adults 18+ with confirmed diagnosis."
                },
                "contactsLocationsModule": {
                    "centralContacts": [
                        {
                            "name": "Dr. Jones",
                            "email": "jones@hospital.org",
                            "phone": "555-1234",
                        }
                    ],
                    "locations": [
                        {
                            "facility": "Hospital A",
                            "city": "Boston",
                            "state": "MA",
                            "country": "US",
                        }
                    ],
                },
            }
        }
        mock_resp = _mock_response(study)
        with patch(_CT_PATCH, return_value=mock_resp):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = get_trial_details.func(
                    nct_id="NCT01234567", runtime=MockRuntime()
                )
                assert "NCT01234567" in result
                assert "COMPLETED" in result
                assert "PHASE3" in result
                assert "This study tests X." in result
                assert "Compound A" in result
                assert "Survival at 6 months" in result
                assert "Quality of life" in result
                assert "Adults 18+" in result
                assert "Dr. Jones" in result
                assert "Hospital A" in result

    def test_empty_protocol_section(self):
        mock_resp = _mock_response({"protocolSection": {}})
        with patch(_CT_PATCH, return_value=mock_resp):
            result = get_trial_details.func(
                nct_id="NCT000", runtime=MockRuntime()
            )
            assert "No data found" in result

    def test_api_error_returns_message(self):
        with patch(_CT_PATCH, side_effect=requests.RequestException("404")):
            result = get_trial_details.func(
                nct_id="NCT999", runtime=MockRuntime()
            )
            assert "Error" in result
            assert "NCT999" in result

    def test_eligibility_truncated_at_1000_chars(self):
        long_text = "X" * 1500
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1", "briefTitle": "T"},
                "statusModule": {},
                "designModule": {},
                "descriptionModule": {},
                "armsInterventionsModule": {},
                "outcomesModule": {},
                "eligibilityModule": {"eligibilityCriteria": long_text},
                "contactsLocationsModule": {},
            }
        }
        mock_resp = _mock_response(study)
        with patch(_CT_PATCH, return_value=mock_resp):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = get_trial_details.func(nct_id="NCT1", runtime=MockRuntime())
                assert "[truncated]" in result


# ---------------------------------------------------------------------------
# search_pubmed
# ---------------------------------------------------------------------------

_PUBMED_SEARCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>99887766</PMID>
            <Article>
                <ArticleTitle>Alzheimer Mechanisms Review</ArticleTitle>
                <Journal>
                    <Title>Cell</Title>
                    <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
                </Journal>
                <AuthorList>
                    <Author><LastName>Brown</LastName><Initials>K</Initials></Author>
                </AuthorList>
                <Abstract>
                    <AbstractText>Detailed abstract content here.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""


class TestSearchPubmed:
    def test_search_params_construction(self):
        search_data = {"esearchresult": {"idlist": []}}
        mock_search = _mock_response(search_data)
        with patch(_CT_PATCH, return_value=mock_search) as mock_get:
            search_pubmed.func(query="Alzheimer", max_results=5, runtime=MockRuntime())
            _url, kwargs = mock_get.call_args
            params = kwargs["params"]
            assert params["db"] == "pubmed"
            assert params["term"] == "Alzheimer"
            assert params["retmax"] == 5
            assert params["retmode"] == "json"

    def test_empty_id_list_returns_message(self):
        mock_resp = _mock_response({"esearchresult": {"idlist": []}})
        with patch(_CT_PATCH, return_value=mock_resp):
            result = search_pubmed.func(
                query="UnknownTerm99", max_results=10, runtime=MockRuntime()
            )
            assert "No PubMed results found" in result
            assert "UnknownTerm99" in result

    def test_search_api_error(self):
        with patch(_CT_PATCH, side_effect=requests.RequestException("timeout")):
            result = search_pubmed.func(
                query="Alzheimer", max_results=10, runtime=MockRuntime()
            )
            assert "Error" in result

    def test_formats_articles_as_markdown(self):
        search_data = {"esearchresult": {"idlist": ["99887766"]}}
        mock_search = _mock_response(search_data)
        mock_fetch = _mock_response({})
        mock_fetch.text = _PUBMED_SEARCH_XML

        with patch(_CT_PATCH, side_effect=[mock_search, mock_fetch]):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = search_pubmed.func(
                    query="Alzheimer", max_results=10, runtime=MockRuntime()
                )
                assert "## PubMed Papers" in result
                assert "99887766" in result
                assert "Alzheimer Mechanisms Review" in result
                assert "Cell, 2023" in result
                assert "Brown K" in result
                assert "### 1." in result

    def test_fetch_api_error_after_search(self):
        search_data = {"esearchresult": {"idlist": ["123"]}}
        mock_search = _mock_response(search_data)
        with patch(
            _CT_PATCH,
            side_effect=[mock_search, requests.RequestException("fetch fail")],
        ):
            result = search_pubmed.func(
                query="Alzheimer", max_results=10, runtime=MockRuntime()
            )
            assert "Error" in result

    def test_batch_fetch_uses_comma_joined_ids(self):
        search_data = {"esearchresult": {"idlist": ["111", "222", "333"]}}
        mock_search = _mock_response(search_data)
        mock_fetch = _mock_response({})
        mock_fetch.text = "<PubmedArticleSet></PubmedArticleSet>"

        with patch(_CT_PATCH, side_effect=[mock_search, mock_fetch]) as mock_get:
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                search_pubmed.func(
                    query="X", max_results=10, runtime=MockRuntime()
                )
                second_call = mock_get.call_args_list[1]
                _url, kwargs = second_call
                assert kwargs["params"]["id"] == "111,222,333"


# ---------------------------------------------------------------------------
# get_pubmed_abstract
# ---------------------------------------------------------------------------


class TestGetPubmedAbstract:
    def test_returns_article_markdown(self):
        mock_resp = _mock_response({})
        mock_resp.text = _PUBMED_SEARCH_XML
        with patch(_CT_PATCH, return_value=mock_resp):
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                result = get_pubmed_abstract.func(
                    pmid="99887766", runtime=MockRuntime()
                )
                assert "99887766" in result
                assert "Alzheimer Mechanisms Review" in result
                assert "Cell, 2023" in result

    def test_pmid_in_fetch_params(self):
        mock_resp = _mock_response({})
        mock_resp.text = _PUBMED_SEARCH_XML
        with patch(_CT_PATCH, return_value=mock_resp) as mock_get:
            with patch(_SR_PATCH, side_effect=_passthrough_send_results):
                get_pubmed_abstract.func(pmid="99887766", runtime=MockRuntime())
                _url, kwargs = mock_get.call_args
                assert kwargs["params"]["id"] == "99887766"
                assert kwargs["params"]["db"] == "pubmed"
                assert kwargs["params"]["retmode"] == "xml"

    def test_no_articles_returns_message(self):
        mock_resp = _mock_response({})
        mock_resp.text = "<PubmedArticleSet></PubmedArticleSet>"
        with patch(_CT_PATCH, return_value=mock_resp):
            result = get_pubmed_abstract.func(pmid="00000", runtime=MockRuntime())
            assert "No article found" in result
            assert "00000" in result

    def test_api_error_returns_message(self):
        with patch(_CT_PATCH, side_effect=requests.RequestException("error")):
            result = get_pubmed_abstract.func(pmid="12345", runtime=MockRuntime())
            assert "Error" in result
            assert "12345" in result

    def test_malformed_xml_returns_message(self):
        mock_resp = _mock_response({})
        mock_resp.text = "not valid xml <<<"
        with patch(_CT_PATCH, return_value=mock_resp):
            result = get_pubmed_abstract.func(pmid="12345", runtime=MockRuntime())
            assert "Error" in result
