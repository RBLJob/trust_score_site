"""
TruPartner Score website (trust-score-site) backed by BigQuery.

Same pages, routes, form fields and template variables as before; the data now comes from the
daily rbl-pipeline tables instead of the Google Sheet:
  trupartner_app_names       name -> entity_id (autocomplete, lookup)
  trupartner_app_partners    one row per publisher with every displayed field
  trupartner_app_brand_perf  per-brand performance (brand table)
  v_hubspot_best_contact     contact name/email (policy-tagged: the app's service account is a reader)
  gemini_lite                classifies a pasted brand URL into an advertiser type

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
SIMILAR_LIMIT = 40
ADV_MIN_SHARE = 0.2       # a publisher "promotes" a type if >= 20% of its commission is from that type

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


# --- shaping rows for the existing templates ------------------------------------------------------

def to_template(p: dict, contact: str | None = None) -> dict:
    return {
        "Clean Up Name": p.get("name"),
        "TRUST SCORE RATING": float(p.get("score") or 0),
        "Website": p.get("website") or "",
        "Associated Contact": contact or "",
        "Description": p.get("description") or "",
        "Industry Vertical": p.get("industry_vertical") or "",
        "Company Business Model": p.get("business_model") or "",
        "Sub Company Vertical": p.get("sub_vertical") or "",
        "RBL Brand": p.get("rbl_brands") or "",
        "Estimated Avg. Ahrefs DR": float(p.get("ahrefs_dr") or 0),
        "Estimated Avg. Moz DA": float(p.get("moz_da") or 0),
        "Estimated Avg. Semrush AS": float(p.get("semrush_as") or 0),
    }


def contacts_for(company_ids: list[str]) -> dict[str, str]:
    ids = [c for c in company_ids if c]
    if not ids:
        return {}
    try:
        rows = q(f"""SELECT company_id, firstname, lastname, email FROM `{D}.v_hubspot_best_contact`
                     WHERE company_id IN UNNEST(@ids)""", ids=ids)
    except Exception as e:   # no fine-grained read access: show pages without contacts
        print(f"[contacts] unavailable: {e}")
        return {}
    out = {}
    for r in rows:
        name = " ".join(x for x in (r.get("firstname"), r.get("lastname")) if x)
        out[r["company_id"]] = f"{name} <{r['email']}>" if name and r.get("email") else (r.get("email") or name)
    return out


PARTNER_COLS = """entity_id, hs_object_id, name, score, website, description, industry_vertical, business_model,
  sub_vertical, rbl_brands, ahrefs_dr, moz_da, semrush_as, publisher_category, top_advertiser_type"""


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
    partner_input = request.form.get("partner_name", "").strip()
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

        # similar partners: same publisher category + promotes the same advertiser type
        # (publishers with no earnings: same industry vertical instead), sorted by TruPartner Score
        similar = q(f"""
            SELECT {PARTNER_COLS} FROM `{D}.trupartner_app_partners` p
            WHERE p.entity_id != @id
              AND (@cat IS NULL OR p.publisher_category = @cat)
              AND (
                (@adv IS NOT NULL AND EXISTS (SELECT 1 FROM UNNEST(p.advertiser_types) t
                                              WHERE t.advertiser_type = @adv AND t.share >= {ADV_MIN_SHARE}))
                OR (@adv IS NULL AND @ind IS NOT NULL AND p.industry_vertical = @ind)
                OR (@adv IS NULL AND @ind IS NULL AND p.business_model = @bm))
            ORDER BY p.score DESC
            LIMIT {SIMILAR_LIMIT}""",
            id=eid, cat=me.get("publisher_category"), adv=me.get("top_advertiser_type"),
            ind=me.get("industry_vertical"), bm=me.get("business_model"))

        contacts = contacts_for([me.get("hs_object_id")] + [s.get("hs_object_id") for s in similar])
        similar_partners = [to_template(s, contacts.get(s.get("hs_object_id"))) for s in similar]
        return render_template("result_trust_score.html",
                               partner_name=me["name"],
                               trust_score=float(me.get("score") or 0),
                               affiliate_brands=brands,
                               similar_partners=similar_partners)
    except Exception as e:
        print(f"[result] {type(e).__name__}: {e}")
        return render_template("result_trust_score.html", error="An unexpected error occurred.")


# --- brand URL lookup ---------------------------------------------------------------------------------

SUB_CATEGORIES = ["Advertising & Marketing", "Apparel & Accessories", "Arts, Entertainment, & Recreation",
    "Automotive / Tools", "Beauty & Personal Care", "Consumer Services", "Education", "E-commerce Tech",
    "Financial Services", "Food & Beverage", "General / Other", "Healthcare", "Health/Wellness", "Home Services",
    "HR & Recruitment", "Industrial & Commercial", "Interior Design / Home Organization", "Lifestyle",
    "Logistics & Freight", "Media & Publishing", "Personal Banking/Taxes", "Personal Development", "Pets",
    "Professional Services", "Real Estate", "Retail & E-commerce", "Sports & Fitness", "Technology & SaaS",
    "Travel & Hospitality", "Wholesale & Retail"]
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


def classify_brand(site_text: str) -> str | None:
    prompt = ("Classify this BRAND (an advertiser) by the market it sells into. Pick exactly ONE sub-category:\n"
              + BRAND_DEFS + '\nReply with JSON only: {"sub_category": "..."}\n\n' + site_text)
    rows = q(f"""SELECT REGEXP_EXTRACT(ml_generate_text_llm_result, r'"sub_category"\\s*:\\s*"([^"]+)"') t
                 FROM ML.GENERATE_TEXT(MODEL `{D}.gemini_lite`, (SELECT @p AS prompt),
                   STRUCT(0.0 AS temperature, 40 AS max_output_tokens, TRUE AS flatten_json_output))""", p=prompt)
    t = rows[0]["t"] if rows else None
    return t if t in SUB_CATEGORIES else None


@app.route("/publisher_lookup", methods=["POST"])
def publisher_lookup():
    brand_url = request.form.get("brand_url", "").strip()
    try:
        text = read_site(brand_url)
        if not text:
            return render_template("result_brand_lookup.html", brand_url=brand_url,
                                   error="Could not read that website. Check the URL and try again.")
        adv = classify_brand(text)
        if not adv:
            return render_template("result_brand_lookup.html", brand_url=brand_url,
                                   error="Could not determine what kind of brand this is.")
        print(f"[publisher_lookup] {brand_url} -> {adv}")
        rows = q(f"""
            SELECT {PARTNER_COLS} FROM `{D}.trupartner_app_partners` p
            WHERE EXISTS (SELECT 1 FROM UNNEST(p.advertiser_types) t
                          WHERE t.advertiser_type = @adv AND t.share >= {ADV_MIN_SHARE})
               OR (ARRAY_LENGTH(p.advertiser_types) IS NULL AND p.sub_vertical = @adv)
            ORDER BY p.score DESC
            LIMIT {SIMILAR_LIMIT}""", adv=adv)
        contacts = contacts_for([r.get("hs_object_id") for r in rows])
        publishers = [to_template(r, contacts.get(r.get("hs_object_id"))) for r in rows]
        if not publishers:
            return render_template("result_brand_lookup.html", brand_url=brand_url,
                                   error=f"No publishers found yet for {adv} brands.")
        return render_template("result_brand_lookup.html", publishers=publishers, brand_url=brand_url)
    except Exception as e:
        print(f"[publisher_lookup] {type(e).__name__}: {e}")
        return render_template("result_brand_lookup.html", brand_url=brand_url, error="An unexpected error occurred.")


# --- CSV download (unchanged contract: POST {"publishers": [...]}) ----------------------------------

CSV_COLUMNS = [("Clean Up Name", "Partner Name"), ("TRUST SCORE RATING", "Trust Partner Score"),
               ("Estimated Avg. Ahrefs DR", "Avg. Ahrefs"), ("Estimated Avg. Moz DA", "Avg. Moz"),
               ("Estimated Avg. Semrush AS", "Avg. Semrush"), ("Website", "URL"), ("Associated Contact", "Contact"),
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
