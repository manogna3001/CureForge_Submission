import json
import xml.etree.ElementTree as ET

import feedparser
import requests
from langchain.tools import ToolRuntime, tool

from app.src.core.state import ResearchAgentState
from app.src.core.tools.phases._shared import get_markdown, send_results
from app.src.utils.logger import get_logger

logger = get_logger(__name__)

_CLINICAL_TRIALS_BASE = "https://clinicaltrials.gov/api/v2"
_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REQUEST_TIMEOUT = 30


def _get_with_retry(url: str, params: dict | None = None) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("Request to %s failed, retrying: %s", url, exc)
    raise last_exc  # type: ignore[misc]


def _format_trial(study: dict) -> str:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    arms = proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    locations_mod = proto.get("contactsLocationsModule", {})

    nct_id = ident.get("nctId", "N/A")
    title = ident.get("briefTitle", "N/A")
    status = status_mod.get("overallStatus", "N/A")
    phases = design.get("phases", [])
    phase_str = ", ".join(phases) if phases else "N/A"

    interventions = arms.get("interventions", [])
    intervention_str = (
        ", ".join(i.get("name", "") for i in interventions if i.get("name")) or "N/A"
    )

    primary_outcomes = outcomes.get("primaryOutcomes", [])
    primary_str = (
        "; ".join(o.get("measure", "") for o in primary_outcomes if o.get("measure"))
        or "N/A"
    )

    locations = locations_mod.get("locations", [])
    location_parts = []
    for loc in locations[:3]:
        parts = [
            p
            for p in [
                loc.get("facility"),
                loc.get("city"),
                loc.get("state"),
                loc.get("country"),
            ]
            if p
        ]
        if parts:
            location_parts.append(", ".join(parts))
    location_str = "; ".join(location_parts) if location_parts else "N/A"

    return (
        f"### {nct_id} - {title}\n"
        f"- **Status**: {status}\n"
        f"- **Phase**: {phase_str}\n"
        f"- **Interventions**: {intervention_str}\n"
        f"- **Primary Outcome**: {primary_str}\n"
        f"- **Locations**: {location_str}"
    )


def _parse_pubmed_article(article_elem: ET.Element) -> dict:
    citation = article_elem.find("MedlineCitation")
    if citation is None:
        return {}

    pmid = citation.findtext("PMID", "")
    article = citation.find("Article")
    if article is None:
        return {"pmid": pmid}

    title = article.findtext("ArticleTitle", "") or ""

    authors = []
    for author in article.findall("AuthorList/Author"):
        last = author.findtext("LastName", "")
        initials = author.findtext("Initials", "")
        if last:
            authors.append(f"{last} {initials}".strip())

    journal = article.findtext("Journal/Title", "") or ""
    year = (
        article.findtext("Journal/JournalIssue/PubDate/Year")
        or article.findtext("Journal/JournalIssue/PubDate/MedlineDate", "")
        or ""
    )

    abstract_parts = []
    for text_elem in article.findall("Abstract/AbstractText"):
        label = text_elem.get("Label")
        content = (text_elem.text or "").strip()
        if label and content:
            abstract_parts.append(f"{label}: {content}")
        elif content:
            abstract_parts.append(content)
    abstract = " ".join(abstract_parts)

    mesh_terms = [
        m.text
        for m in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
        if m.text
    ]

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "abstract": abstract,
        "mesh_terms": mesh_terms,
    }


def _format_pubmed_article(data: dict) -> str:
    pmid = data.get("pmid", "N/A")
    title = data.get("title") or "N/A"
    authors = data.get("authors", [])
    author_str = ", ".join(authors[:6])
    if len(authors) > 6:
        author_str += " et al."
    journal = data.get("journal") or "N/A"
    year = data.get("year", "")
    journal_str = f"{journal}, {year}" if year else journal
    abstract = data.get("abstract") or "N/A"
    mesh_terms = data.get("mesh_terms", [])

    lines = [
        f"### PMID: {pmid} - {title}",
        f"- **Authors**: {author_str or 'N/A'}",
        f"- **Journal**: {journal_str}",
        f"- **Abstract**: {abstract}",
    ]
    if mesh_terms:
        lines.append(f"- **MeSH Terms**: {', '.join(mesh_terms[:10])}")
    return "\n".join(lines)


@tool
def research_scan_literature(
    query: str,
    max_results: int,
    runtime: ToolRuntime[None, ResearchAgentState],
) -> str:
    """Search arXiv for biomedical papers. Returns JSON list of papers with title, authors, date, summary, PDF link.

    Use in research phase to gather evidence.
    Results over max_results_length saved to file; read with read_file.

    Args:
        query: Search term (lowercase single word).
        max_results: Max papers to retrieve (int).
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        entry_dict = {
            "title": entry.title,
            "published": entry.published,
            "updated": entry.updated,
            "summary": entry.summary,
            "authors": [a.name for a in entry.authors],
            "links": [
                {"href": link.href}
                for link in entry.links
                if link.type == "application/pdf"
            ],
        }
        results.append(entry_dict)
    results_str = json.dumps(results, indent=2)
    return send_results(str(results_str), runtime)


@tool
def fetch_paper_from_link(
    link: str,
    runtime: ToolRuntime[None, ResearchAgentState],
):
    """Fetch paper PDF content and convert to markdown. Returns text.

    Use after research_scan_literature to read specific papers.
    Multiple fetches: call tool sequentially.

    Args:
        link: PDF URL from research_scan_literature result (str).
    """
    return get_markdown(link, runtime)


@tool
def search_clinical_trials(
    condition: str,
    status: str | None,
    phase: str | None,
    max_results: int,
    runtime: ToolRuntime[None, ResearchAgentState],
) -> str:
    """Search ClinicalTrials.gov for trials matching a condition. Returns markdown list of trials with status, phase, interventions, and outcomes.

    Use in research phase to find real-world clinical trial data.
    Results over max_results_length saved to file; read with read_file.

    Args:
        condition: Disease or condition name (str).
        status: Filter by overall status e.g. RECRUITING, COMPLETED; pass None for any (str | None).
        phase: Filter by phase e.g. PHASE2, PHASE3; pass None for any (str | None).
        max_results: Max trials to retrieve, default 10 (int).
    """
    params: dict = {
        "query.cond": condition,
        "pageSize": max_results,
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,InterventionName,InterventionType,"
            "PrimaryOutcomeMeasure,LocationFacility,LocationCity,LocationState,LocationCountry"
        ),
    }
    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["filter.phase"] = phase

    try:
        resp = _get_with_retry(f"{_CLINICAL_TRIALS_BASE}/studies", params=params)
        data = resp.json()
    except requests.RequestException as exc:
        return f"Error querying ClinicalTrials.gov: {exc}"
    except Exception as exc:
        return f"Unexpected error querying ClinicalTrials.gov: {exc}"

    studies = data.get("studies", [])
    if not studies:
        return f"No clinical trials found for condition: {condition}"

    lines = ["## Clinical Trials\n"]
    for i, study in enumerate(studies, 1):
        trial_md = _format_trial(study)
        lines.append(trial_md.replace("### ", f"### {i}. ", 1))
        lines.append("\n---")

    return send_results("\n".join(lines), runtime)


@tool
def get_trial_details(
    nct_id: str,
    runtime: ToolRuntime[None, ResearchAgentState],
) -> str:
    """Fetch full details for a ClinicalTrials.gov trial by NCT ID. Returns comprehensive trial info in markdown.

    Use after search_clinical_trials to retrieve complete details for a specific trial.

    Args:
        nct_id: Trial identifier e.g. NCT01234567 (str).
    """
    try:
        resp = _get_with_retry(f"{_CLINICAL_TRIALS_BASE}/studies/{nct_id}")
        study = resp.json()
    except requests.RequestException as exc:
        return f"Error fetching trial {nct_id}: {exc}"
    except Exception as exc:
        return f"Unexpected error fetching trial {nct_id}: {exc}"

    proto = study.get("protocolSection", {})
    if not proto:
        return f"No data found for trial: {nct_id}"

    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    arms = proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    eligibility = proto.get("eligibilityModule", {})
    contacts_locs = proto.get("contactsLocationsModule", {})
    desc = proto.get("descriptionModule", {})

    title = ident.get("briefTitle", "N/A")
    official_title = ident.get("officialTitle", "")
    status = status_mod.get("overallStatus", "N/A")
    phases = design.get("phases", [])
    phase_str = ", ".join(phases) if phases else "N/A"
    brief_summary = (desc.get("briefSummary", "") or "").strip()

    interventions = arms.get("interventions", [])
    intervention_lines = [
        f"  - {i.get('name', '')} ({i.get('type', '')})"
        for i in interventions
        if i.get("name")
    ]

    primary_outcomes = outcomes.get("primaryOutcomes", [])
    secondary_outcomes = outcomes.get("secondaryOutcomes", [])

    eligibility_text = eligibility.get("eligibilityCriteria", "N/A") or "N/A"
    if len(eligibility_text) > 1000:
        eligibility_text = eligibility_text[:1000] + "... [truncated]"

    central_contacts = contacts_locs.get("centralContacts", [])
    locations = contacts_locs.get("locations", [])

    lines = [f"## Trial: {nct_id} - {title}"]
    if official_title and official_title != title:
        lines.append(f"**Official Title**: {official_title}\n")
    lines += [
        f"- **Status**: {status}",
        f"- **Phase**: {phase_str}",
    ]
    if brief_summary:
        lines.append(f"\n**Summary**: {brief_summary}")

    if intervention_lines:
        lines.append("\n**Interventions**:")
        lines.extend(intervention_lines)

    if primary_outcomes:
        lines.append("\n**Primary Outcomes**:")
        for o in primary_outcomes:
            measure = o.get("measure", "")
            if measure:
                lines.append(f"  - {measure}")

    if secondary_outcomes:
        lines.append("\n**Secondary Outcomes**:")
        for o in secondary_outcomes[:5]:
            measure = o.get("measure", "")
            if measure:
                lines.append(f"  - {measure}")

    lines.append(f"\n**Eligibility Criteria**:\n{eligibility_text}")

    if central_contacts:
        lines.append("\n**Central Contacts**:")
        for c in central_contacts[:3]:
            contact_parts = [
                p for p in [c.get("name"), c.get("phone"), c.get("email")] if p
            ]
            if contact_parts:
                lines.append(f"  - {', '.join(contact_parts)}")

    if locations:
        lines.append(f"\n**Locations** ({len(locations)} total):")
        for loc in locations[:5]:
            parts = [
                p
                for p in [
                    loc.get("facility"),
                    loc.get("city"),
                    loc.get("state"),
                    loc.get("country"),
                ]
                if p
            ]
            if parts:
                lines.append(f"  - {', '.join(parts)}")
        if len(locations) > 5:
            lines.append(f"  ... and {len(locations) - 5} more")

    return send_results("\n".join(lines), runtime)


@tool
def search_pubmed(
    query: str,
    max_results: int,
    runtime: ToolRuntime[None, ResearchAgentState],
) -> str:
    """Search PubMed for biomedical papers. Returns markdown list of papers with title, authors, journal, and abstract.

    Use in research phase to find peer-reviewed literature.
    Results over max_results_length saved to file; read with read_file.

    Args:
        query: Search query, supports PubMed syntax e.g. "Alzheimer AND treatment" (str).
        max_results: Max papers to retrieve, default 10 (int).
    """
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    try:
        search_resp = _get_with_retry(
            f"{_PUBMED_BASE}/esearch.fcgi", params=search_params
        )
        search_data = search_resp.json()
    except requests.RequestException as exc:
        return f"Error searching PubMed: {exc}"
    except Exception as exc:
        return f"Unexpected error searching PubMed: {exc}"

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return f"No PubMed results found for query: {query}"

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    try:
        fetch_resp = _get_with_retry(
            f"{_PUBMED_BASE}/efetch.fcgi", params=fetch_params
        )
        root = ET.fromstring(fetch_resp.text)
    except requests.RequestException as exc:
        return f"Error fetching PubMed abstracts: {exc}"
    except ET.ParseError as exc:
        return f"Error parsing PubMed response: {exc}"
    except Exception as exc:
        return f"Unexpected error fetching PubMed abstracts: {exc}"

    articles = root.findall("PubmedArticle")
    if not articles:
        return f"No article data returned for query: {query}"

    lines = ["## PubMed Papers\n"]
    for i, article_elem in enumerate(articles, 1):
        data = _parse_pubmed_article(article_elem)
        if not data:
            continue
        article_md = _format_pubmed_article(data)
        lines.append(article_md.replace("### ", f"### {i}. ", 1))
        lines.append("\n---")

    return send_results("\n".join(lines), runtime)


@tool
def get_pubmed_abstract(
    pmid: str,
    runtime: ToolRuntime[None, ResearchAgentState],
) -> str:
    """Fetch full abstract and metadata for a PubMed paper by PMID. Returns detailed paper info in markdown.

    Use after search_pubmed to retrieve complete details for a specific paper.

    Args:
        pmid: PubMed ID e.g. 12345678 (str).
    """
    fetch_params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    try:
        resp = _get_with_retry(f"{_PUBMED_BASE}/efetch.fcgi", params=fetch_params)
        root = ET.fromstring(resp.text)
    except requests.RequestException as exc:
        return f"Error fetching PMID {pmid}: {exc}"
    except ET.ParseError as exc:
        return f"Error parsing PubMed response for PMID {pmid}: {exc}"
    except Exception as exc:
        return f"Unexpected error fetching PMID {pmid}: {exc}"

    articles = root.findall("PubmedArticle")
    if not articles:
        return f"No article found for PMID: {pmid}"

    data = _parse_pubmed_article(articles[0])
    if not data:
        return f"Could not parse article for PMID: {pmid}"

    return send_results(_format_pubmed_article(data), runtime)
