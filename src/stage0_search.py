"""Stage 0: PubMed literature search and DOI-based deduplication."""
import time
import csv
from typing import Optional
from Bio import Entrez
from src.config import ENTREZ_EMAIL, ENTREZ_API_KEY, DATA_DIR

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY


def build_query(substance_name: str) -> str:
    """Build a PubMed query for a substance + pig/swine/piglet keywords."""
    pig_terms = "(pig OR swine OR piglet OR pigs OR piglets)"
    return f'"{substance_name}"[tiab] AND {pig_terms}'


def dedup_by_doi(new_results: list[dict], existing_dois: set[str]) -> tuple[list[dict], list[dict]]:
    """Merge new results with existing DOIs.

    Returns: (merged_full_list, novel_only)
    merged_full_list = all unique articles across new_results and existing_dois
    novel_only = articles in new_results whose DOI is not in existing_dois
    """
    seen = set(existing_dois)
    novel = []
    merged = []

    for r in new_results:
        doi = r.get("doi", "")
        if not doi:
            continue
        if doi not in seen:
            seen.add(doi)
            novel.append(r)
        merged.append(r)

    # Include existing DOIs not already covered by a new_results item
    if merged:
        merged_dois = {r.get("doi", "") for r in merged}
        for doi in existing_dois:
            if doi not in merged_dois:
                merged.append({"doi": doi})

    return merged, novel


def search_substance(substance: str, retmax: int = 5000) -> list[dict]:
    """Search PubMed for a substance, return list of article metadata dicts.

    Uses NCBI E-utilities with history server for efficiency.
    Returns [] on error or empty results.
    """
    query = build_query(substance)
    results = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, usehistory="y")
        search_res = Entrez.read(handle)
        handle.close()

        id_list = search_res.get("IdList", [])
        if not id_list:
            return results

        # Fetch metadata in batches of 200 (NCBI limit per efetch call)
        for i in range(0, len(id_list), 200):
            batch = id_list[i:i + 200]
            time.sleep(0.35)  # NCBI rate limit: ~3 requests/sec without API key
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
            articles = Entrez.read(handle)
            handle.close()

            for article in articles.get("PubmedArticle", []):
                medline = article.get("MedlineCitation", {})
                article_data = medline.get("Article", {})
                pmid = str(medline.get("PMID", ""))

                # Extract DOI from ELocationID
                doi = ""
                eids = article_data.get("ELocationID", [])
                if isinstance(eids, list):
                    for eid in eids:
                        if hasattr(eid, 'attributes') and eid.attributes.get('EIdType') == 'doi':
                            doi = str(eid)
                            break

                title = str(article_data.get("ArticleTitle", ""))
                journal_info = article_data.get("Journal", {})
                journal = str(journal_info.get("Title", ""))

                # Publication year
                pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
                year = None
                y = pub_date.get("Year", "")
                if y:
                    try:
                        year = int(y)
                    except ValueError:
                        pass

                results.append({
                    "doi": doi,
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "publication_year": year,
                    "search_term": substance,
                })
    except Exception as e:
        print(f"Error searching '{substance}': {e}")
    return results


def run_stage0(
    alternative_tsv: str = "ALTERNATIVE.tsv",
    existing_doi_list: Optional[str] = None,
    output_path: str = "data/literature_pool.tsv",
) -> str:
    """Full Stage 0: search all substances from ALTERNATIVE.tsv, dedup, write pool TSV.

    Returns path to the output TSV file.
    """
    from src.glossary import GlossaryIndex

    gi = GlossaryIndex()
    gi.load(alternative_tsv)

    # Collect all unique substance names
    all_substances = sorted(set(
        info["standard_name"] for info in gi._exact.values()
    ))

    # Load existing DOIs if provided
    existing_dois = set()
    if existing_doi_list:
        try:
            with open(existing_doi_list, "r") as f:
                for line in f:
                    doi = line.strip().split("\t")[0] if "\t" in line else line.strip()
                    if doi:
                        existing_dois.add(doi)
        except FileNotFoundError:
            pass

    all_novel = []
    for substance in all_substances:
        results = search_substance(substance)
        _, novel = dedup_by_doi(results, existing_dois)
        all_novel.extend(novel)
        # Track seen DOIs for cross-substance dedup
        for r in results:
            if r.get("doi"):
                existing_dois.add(r["doi"])
        print(f"  {substance}: {len(results)} hits, {len(novel)} novel")

    # Ensure output directory exists
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write pool
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doi", "pmid", "title", "journal", "publication_year", "search_term"],
            delimiter="\t",
        )
        writer.writeheader()
        for r in all_novel:
            writer.writerow(r)

    print(f"\nTotal novel articles in pool: {len(all_novel)}")
    return output_path
