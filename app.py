"""
TruPartner Score website (trust-score-site) backed by BigQuery.

Same pages and routes as before; the data comes from the daily rbl-pipeline tables:
  trupartner_app_names       name -> entity_id (autocomplete, lookup)
  trupartner_app_partners    one row per publisher with every displayed field
  trupartner_app_brand_perf  per-brand performance (brand table)
  brand_advertiser_type      advertiser type of each RBL brand
  v_hubspot_best_contact     contact name/email (policy-tagged: the app's service account is a reader)
  gemini_lite                classifies a pasted brand URL (company vertical + industry)

Both result pages have filters (Industry Vertical, Business Model, Company Vertical), pre-filled
automatically, and every result carries a "why it matched" label:
  Proven       earned on brands of that company vertical (share >= 20% of its commission)
  Topic match  its own website is classified in that company vertical

Credentials: env GCP_SA_JSON (service account key JSON, set in Render) or local gcloud login.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, make_response, render_template, request
from google.cloud import bigquery

PROJECT = "rbl-data-connection"
D = f"{PROJECT}.funnel_export_rblaffiliate"
NAMES_TTL = 3600          # reload the autocomplete list hourly (data refreshes daily)
RESULT_LIMIT = 40
ADV_MIN_SHARE = 0.2       # "proven" = at least 20% of a publisher's commission is from that type of brand

app = Flask(__name__)


def _client() -> bigquery.Client:
    raw = os.environ.get("GCP_SA_JSON")
    if raw:
        from google.oauth2 import service_account
        info = json.loads(raw)
        return bigquery.Client(project=PROJECT,
                               credentials=service_account.Credentials.from_service_account_info(info))
    return bigquery.Client(project=PROJECT)


bq = _client()
_names = {"by_lc": {}, "list": [], "loaded": 0.0}
_names_lock = threading.Lock()


def q(sql: str, **params) -> list[dict]:
    qp = []
    for k, v in params.items():
        if isinstance(v, list):
            qp.append(bigquery.ArrayQueryParameter(k, "STRING", v))
        else:
            qp.append(bigquery.ScalarQueryParameter(k, "STRING", v))
    rows = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=qp)).result()
    return [dict(r.items()) for r in rows]


def names() -> dict:
    with _names_lock:
        if time.time() - _names["loaded"] > NAMES_TTL or not _names["list"]:
            rows = q(f"""SELECT n.name, n.name_lc, n.entity_id, p.score, p.name AS display
                         FROM `{D}.trupartner_app_names` n
                         JOIN `{D}.trupartner_app_partners` p USING (entity_id)""")
            by_lc = {}
            for r in rows:   # if two entities share a name, keep the higher-scored one
                cur = by_lc.get(r["name_lc"])
                if cur is None or (r["score"] or 0) > (cur["score"] or 0):
                    by_lc[r["name_lc"]] = r
            _names.update(by_lc=by_lc, list=sorted(by_lc.values(), key=lambda r: -(r["score"] or 0)),
                          loaded=time.time())
        return _names


# --- framework option lists (filters) -------------------------------------------------------------

INDUSTRIES = ["B2C", "B2B", "B2B / B2C"]
BUSINESS_MODELS = ["Content Creator", "Content Publisher", "Product Review Sites", "Coupons / Deal Sites",
    "Card Linked Offer (CLO) Partners", "Marketing Agency", "SEM / Paid Media Agencies", "SEO Agencies",
    "Technology & SaaS", "AI / Tech Solutions Providers", "Tech Integrators / IT Consultants", "Affiliate Networks",
    "Business Consultancy Firms", "E-learning Platforms", "Merchant Seller / Shopify Merchants", "Wholesale Merchant",
    "Online Communities", "System Internal / Data Privacy", "Other"]
SUB_CATEGORIES = ["Advertising & Marketing", "Apparel & Accessories", "Arts, Entertainment, & Recreation",
    "Automotive / Tools", "Beauty & Personal Care", "Consumer Services", "Education", "E-commerce Tech",
    "Financial Services", "Food & Beverage", "General / Other", "Healthcare", "Health/Wellness", "Home Services",
    "HR & Recruitment", "Industrial & Commercial", "Interior Design / Home Organization", "Lifestyle",
    "Logistics & Freight", "Media & Publishing", "Personal Banking/Taxes", "Personal Development", "Pets",
    "Professional Services", "Real Estate", "Retail & E-commerce", "Sports & Fitness", "Technology & SaaS",
    "Travel & Hospitality", "Wholesale & Retail"]
OPTIONS = {"industry": INDUSTRIES, "business_model": BUSINESS_MODELS, "company_vertical": SUB_CATEGORIES}


def _pick(value: str | None, allowed: list[str]) -> str | None:
    return value if value in allowed else None


# --- shaping rows for the templates ---------------------------------------------------------------

def to_template(p: dict, contact: dict | None = None) -> dict:
    contact = contact or {}
    return {
        "Clean Up Name": p.get("name"),
        "TRUST SCORE RATING": float(p.get("score") or 0),
        "Website": p.get("website") or "",
        "Contact Name": contact.get("name") or "",
        "Contact Title": contact.get("title") or "",
        "Contact Email": contact.get("email") or "",
        "Description": p.get("description") or "",
        "Industry Vertical": p.get("industry_vertical") or "",
        "Company Business Model": p.get("business_model") or "",
        "Sub Company Vertical": p.get("sub_vertical") or "",
        "RBL Brand": p.get("rbl_brands") or "",
        "Estimated Avg. Ahrefs DR": float(p.get("ahrefs_dr") or 0),
        "Estimated Avg. Moz DA": float(p.get("moz_da") or 0),
        "Estimated Avg. Semrush AS": float(p.get("semrush_as") or 0),
        "Match": p.get("match_label") or "",
        "Match Detail": p.get("match_detail") or "",
    }


def contacts_for(company_ids: list[str]) -> dict[str, dict]:
    ids = [c for c in company_ids if c]
    if not ids:
        return {}
    try:
        rows = q(f"""SELECT company_id, firstname, lastname, email, jobtitle FROM `{D}.v_hubspot_best_contact`
                     WHERE company_id IN UNNEST(@ids)""", ids=ids)
    except Exception as e:   # no fine-grained read access: show pages without contacts
        print(f"[contacts] unavailable: {e}")
        return {}
    out = {}
    for r in rows:
        name = " ".join(x for x in (r.get("firstname"), r.get("lastname")) if x)
        out[r["company_id"]] = {"name": name, "title": (r.get("jobtitle") or "").strip(), "email": r.get("email") or ""}
    return out


PARTNER_COLS = """entity_id, hs_object_id, name, score, website, description, industry_vertical, business_model,
  sub_vertical, rbl_brands, ahrefs_dr, moz_da, semrush_as, publisher_category, top_advertiser_type, advertiser_types"""


def find_publishers(cv: str | None, ind: str | None, bm: str | None,
                    exclude_entity: str | None = None, hide_brand: str | None = None) -> list[dict]:
    """Publishers matching the filters, best TruPartner Score first.
    cv  company vertical: publisher is Proven (earned on that type of brand) or a Topic match (its own site)
    ind industry vertical: B2B or B2C also include 'B2B / B2C' (dual-audience media groups)
    bm  business model (partner type)"""
    return q(f"""
        WITH pb AS (
          SELECT bp.entity_id, STRING_AGG(bp.Affiliate_Brand, ', ' ORDER BY bp.total_commission DESC LIMIT 3) brands
          FROM `{D}.trupartner_app_brand_perf` bp
          JOIN `{D}.brand_advertiser_type` bt ON bt.brand = bp.Affiliate_Brand
          WHERE bt.advertiser_type = @cv AND bp.total_commission > 0
          GROUP BY 1),
        base AS (
          SELECT {PARTNER_COLS},
            EXISTS (SELECT 1 FROM UNNEST(p.advertiser_types) t
                    WHERE t.advertiser_type = @cv AND t.share >= {ADV_MIN_SHARE}) AS proven,
            COALESCE(p.sub_vertical = @cv, FALSE) AS topic
          FROM `{D}.trupartner_app_partners` p
          WHERE (@ex IS NULL OR p.entity_id != @ex)
            AND (@ind IS NULL OR p.industry_vertical IN (@ind, 'B2B / B2C'))
            AND (@bm IS NULL OR p.business_model = @bm)
            AND (@hide IS NULL OR p.entity_id NOT IN (
                  SELECT entity_id FROM `{D}.trupartner_app_brand_perf`
                  WHERE Affiliate_Brand = @hide AND total_commission > 0)))
        SELECT b.*,
          CASE WHEN @cv IS NULL THEN '' WHEN proven THEN 'Proven' WHEN topic THEN 'Topic match' END AS match_label,
          CASE WHEN @cv IS NULL THEN ''
               WHEN proven THEN CONCAT('Earned on ', COALESCE(pb.brands, @cv))
               WHEN topic THEN CONCAT('Website is about ', @cv) END AS match_detail
        FROM base b LEFT JOIN pb USING (entity_id)
        WHERE @cv IS NULL OR proven OR topic
        ORDER BY score DESC
        LIMIT {RESULT_LIMIT}""", cv=cv, ind=ind, bm=bm, ex=exclude_entity, hide=hide_brand)


def shape(rows: list[dict]) -> list[dict]:
    contacts = contacts_for([r.get("hs_object_id") for r in rows])
    return [to_template(r, contacts.get(r.get("hs_object_id"))) for r in rows]


# --- routes -----------------------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("query", "").strip().lower()
    if len(query) < 2:
        return jsonify([])
    try:
        rows = names()["list"]   # already sorted by score, best first
        hits = [r for r in rows if r["name_lc"].startswith(query)] + \
               [r for r in rows if query in r["name_lc"] and not r["name_lc"].startswith(query)]
        seen, out = set(), []    # one suggestion per publisher, shown under its main name
        for r in hits:
            if r["entity_id"] not in seen:
                seen.add(r["entity_id"])
                out.append(r["display"] or r["name"])
            if len(out) == 20:
                break
        return jsonify(out)
    except Exception as e:
        print(f"[autocomplete] {e}")
        return jsonify([])


@app.route("/result", methods=["POST"])
def result():
    f = request.form
    partner_input = f.get("partner_name", "").strip()
    try:
        hit = names()["by_lc"].get(partner_input.lower())
        if not hit:
            return render_template("result_trust_score.html",
                                   error=f"Partner '{partner_input}' not found. Pick a name from the suggestions.")
        eid = hit["entity_id"]
        me = q(f"SELECT {PARTNER_COLS} FROM `{D}.trupartner_app_partners` WHERE entity_id = @id", id=eid)
        if not me:
            return render_template("result_trust_score.html", error="No data found for the selected partner.")
        me = me[0]

        brands = q(f"""SELECT Affiliate_Brand, total_revenue, total_order_action, total_clicks, COALESCE(CR, 0) CR
                       FROM `{D}.trupartner_app_brand_perf` WHERE entity_id = @id
                       ORDER BY total_commission DESC""", id=eid)

        if f.get("filtered"):
            cv = _pick(f.get("company_vertical"), SUB_CATEGORIES)
            ind = _pick(f.get("industry"), INDUSTRIES)
            bm = _pick(f.get("business_model"), BUSINESS_MODELS)
        else:   # pre-fill from the partner itself
            cv = me.get("top_advertiser_type") or _pick(me.get("sub_vertical"), SUB_CATEGORIES)
            ind = _pick(me.get("industry_vertical"), INDUSTRIES)
            bm = _pick(me.get("business_model"), BUSINESS_MODELS)

        similar = shape(find_publishers(cv, ind, bm, exclude_entity=eid))
        filters = dict(action="/result", hidden={"partner_name": me["name"]},
                       industry=ind or "", business_model=bm or "", company_vertical=cv or "",
                       options=OPTIONS, show_hide_current=False, hide_current=False, current_brand="",
                       note=f"Pre-filled from {me['name']}. Change the filters and click Update results.")
        return render_template("result_trust_score.html",
                               partner_name=me["name"],
                               trust_score=float(me.get("score") or 0),
                               affiliate_brands=brands,
                               similar_partners=similar,
                               filters=filters)
    except Exception as e:
        print(f"[result] {type(e).__name__}: {e}")
        return render_template("result_trust_score.html", error="An unexpected error occurred.")


# --- brand URL lookup ---------------------------------------------------------------------------------

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def read_site(url: str) -> str | None:
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12, allow_redirects=True)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text[:600_000], "html.parser")
        for t in soup(["script", "style", "noscript", "svg"]):
            t.decompose()
        meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        heads = " | ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])[:15])
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        body = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:1500]
        return f"Website: {url}\nTitle: {title}\nDescription: {meta.get('content', '') if meta else ''}\nHeadings: {heads}\nPage text: {body}"
    except Exception as e:
        print(f"[read_site] {e}")
        return None


BRAND_DEFS = """- Advertising & Marketing: agencies, networks, ad tech, PPC, SEO, affiliate management, media buying
- Apparel & Accessories: clothing, fashion, jewelry, footwear, accessories
- Arts, Entertainment, & Recreation: video games, entertainment, music, film, events, tickets, hobbies
- Automotive / Tools: cars, auto parts, repair, garage tools, vehicle accessories
- Beauty & Personal Care: cosmetics, skincare, haircare, grooming
- Consumer Services: consumer utilities, moving, insurance, warranties, local services
- Education: tutoring, courses, test prep, language learning
- E-commerce Tech: tools for online sellers, Amazon FBA, listing optimization
- Financial Services: business financing, corporate banking, payment processing, merchant loans
- Food & Beverage: food, drinks, wine, meal kits, groceries, restaurants
- General / Other: none of the others fits
- Healthcare: clinical care, therapy, mental health, telehealth, pharmacy
- Health/Wellness: supplements, nutrition, sleep and self-care products, holistic wellness
- Home Services: pest control, cleaning, lawn care, HVAC, home maintenance
- HR & Recruitment: hiring, staffing, payroll, Employer of Record, HR software
- Industrial & Commercial: manufacturing, raw materials, industrial equipment
- Interior Design / Home Organization: furniture, mattresses, bedding, home decor, storage
- Lifestyle: general lifestyle and family products
- Logistics & Freight: shipping, freight, fulfillment, warehousing
- Media & Publishing: news, magazines, streaming media subscriptions
- Personal Banking/Taxes: consumer banking, credit cards, investing apps, budgeting, tax filing
- Personal Development: coaching, productivity, self-improvement
- Pets: pet food, pet pharmacy, pet products
- Professional Services: legal, accounting, business formation, consulting
- Real Estate: property, mortgages, rentals of homes
- Retail & E-commerce: multi-category retailers and marketplaces
- Sports & Fitness: fitness equipment, gyms, coaching apps, sporting goods, outdoor gear
- Technology & SaaS: software, apps, consumer electronics, AI tools
- Travel & Hospitality: travel booking, hotels, flights, RV and car rental, luggage
- Wholesale & Retail: wholesale, bulk, liquidation"""


def classify_brand(site_text: str) -> tuple[str | None, str | None]:
    """(company vertical, industry) of a brand from its homepage."""
    prompt = ("Classify this BRAND (an advertiser). Pick exactly ONE sub-category for the market it sells into:\n"
              + BRAND_DEFS
              + "\nAnd its industry: B2C if it sells to consumers, B2B if it sells to businesses.\n"
              + 'Reply with JSON only: {"sub_category": "...", "industry": "B2C or B2B"}\n\n' + site_text)
    rows = q(f"""SELECT REGEXP_EXTRACT(ml_generate_text_llm_result, r'"sub_category"\\s*:\\s*"([^"]+)"') sub,
                        REGEXP_EXTRACT(ml_generate_text_llm_result, r'"industry"\\s*:\\s*"([^"]+)"') ind
                 FROM ML.GENERATE_TEXT(MODEL `{D}.gemini_lite`, (SELECT @p AS prompt),
                   STRUCT(0.0 AS temperature, 60 AS max_output_tokens, TRUE AS flatten_json_output))""", p=prompt)
    if not rows:
        return None, None
    return _pick(rows[0]["sub"], SUB_CATEGORIES), _pick(rows[0]["ind"], ["B2C", "B2B"])


def current_brand(url: str) -> str | None:
    """RBL brand name if this URL is one of RBL's brands (used for 'hide current partners')."""
    rows = q(f"""SELECT brand FROM `{D}.brand_advertiser_type`
                 WHERE website IS NOT NULL
                   AND `{D}.fn_site_key`(website) = `{D}.fn_site_key`(@u)
                 LIMIT 1""", u=url)
    return rows[0]["brand"] if rows else None


@app.route("/publisher_lookup", methods=["POST"])
def publisher_lookup():
    f = request.form
    brand_url = f.get("brand_url", "").strip()
    try:
        if f.get("filtered"):
            cv = _pick(f.get("company_vertical"), SUB_CATEGORIES)
            ind = _pick(f.get("industry"), INDUSTRIES)
            bm = _pick(f.get("business_model"), BUSINESS_MODELS)
            detected = f.get("detected", "")
            hide = f.get("hide_current") == "1"
        else:
            text = read_site(brand_url)
            if not text:
                return render_template("result_brand_lookup.html", brand_url=brand_url,
                                       error="Could not read that website. Check the URL and try again.")
            cv, ind = classify_brand(text)
            if not cv:
                return render_template("result_brand_lookup.html", brand_url=brand_url,
                                       error="Could not determine what kind of brand this is.")
            bm, hide = None, True
            detected = cv + (f" · {ind}" if ind else "")
            print(f"[publisher_lookup] {brand_url} -> {detected}")

        brand = current_brand(brand_url)
        publishers = shape(find_publishers(cv, ind, bm, hide_brand=brand if (hide and brand) else None))
        note = f"Detected for this brand: {detected}. " if detected else ""
        if brand:
            note += f"{brand} is an RBL brand. "
        note += "Change the filters and click Update results."
        filters = dict(action="/publisher_lookup", hidden={"brand_url": brand_url, "detected": detected},
                       industry=ind or "", business_model=bm or "", company_vertical=cv or "",
                       options=OPTIONS, show_hide_current=bool(brand), hide_current=hide,
                       current_brand=brand or "", note=note)
        return render_template("result_brand_lookup.html", publishers=publishers, brand_url=brand_url,
                               filters=filters)
    except Exception as e:
        print(f"[publisher_lookup] {type(e).__name__}: {e}")
        return render_template("result_brand_lookup.html", brand_url=brand_url, error="An unexpected error occurred.")


# --- CSV download (contract: POST {"publishers": [...]}) -------------------------------------------

CSV_COLUMNS = [("Clean Up Name", "Partner Name"), ("TRUST SCORE RATING", "Trust Partner Score"),
               ("Match", "Match"), ("Match Detail", "Why"),
               ("Estimated Avg. Ahrefs DR", "Avg. Ahrefs"), ("Estimated Avg. Moz DA", "Avg. Moz"),
               ("Estimated Avg. Semrush AS", "Avg. Semrush"), ("Website", "URL"), ("Contact Name", "Contact Name"),
               ("Contact Title", "Title"), ("Contact Email", "Email"),
               ("Description", "Description"), ("Industry Vertical", "Industry Vertical"),
               ("Company Business Model", "Business Model"), ("Sub Company Vertical", "Company Vertical"),
               ("RBL Brand", "RBL Brand")]


@app.route("/download_csv", methods=["POST"])
def download_csv():
    data = (request.json or {}).get("publishers", [])
    if not data:
        return jsonify({"error": "No data provided"}), 400
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([label for _, label in CSV_COLUMNS])
    for row in data:
        w.writerow([row.get(key, "") for key, _ in CSV_COLUMNS])
    resp = make_response(out.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=similar_partners.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/health")
def health():
    n = names()
    return jsonify({"ok": True, "names": len(n["list"]), "names_loaded_at": n["loaded"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
