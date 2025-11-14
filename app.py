from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import gspread
from google.oauth2 import service_account
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import re
import numpy as np
import os
import json
from datetime import datetime, timedelta
from langdetect import detect, LangDetectException

app = Flask(__name__)

# -------------------------------
# Configuration
# -------------------------------
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1b4_yuhEeLN-u21KHLEJOenDa1EdG6iQXpUi8ASXYZHk/edit?gid=1170846120'
WORKSHEET_NAME = 'Render Upload'

# Cache configuration - ENHANCED with TF-IDF caching
CACHE_DURATION = timedelta(minutes=30)
cache_data = {
    'data': None,
    'timestamp': None,
    'partner_names': None,
    'tfidf_vectorizer': None,
    'tfidf_matrix': None,
    'filtered_df': None
}

# -------------------------------
# Helper Functions
# -------------------------------

def get_google_credentials():
    """Get Google credentials from environment variable or file"""
    try:
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            print("✅ Using credentials from environment variable")
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            )
        else:
            print("✅ Using credentials from credentials.json file")
            credentials = service_account.Credentials.from_service_account_file(
                "credentials.json",
                scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            )
        return credentials
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        raise

def is_cache_valid():
    if (cache_data['data'] is None or 
        cache_data['timestamp'] is None or 
        cache_data['tfidf_vectorizer'] is None or 
        cache_data['tfidf_matrix'] is None or
        cache_data['filtered_df'] is None):
        return False
    time_elapsed = datetime.now() - cache_data['timestamp']
    return time_elapsed < CACHE_DURATION

def is_english_text(text):
    if not text or len(text.strip()) < 10:
        return False
    try:
        detected_lang = detect(text)
        is_eng = detected_lang == 'en'
        if not is_eng:
            print(f"🌐 Non-English text detected: {detected_lang}")
        return is_eng
    except LangDetectException:
        latin_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text.strip())
        return total_chars > 0 and (latin_chars / total_chars) >= 0.7

def resolve_shortened_url(url, timeout=10):
    shortened_domains = ['t.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'buff.ly']
    try:
        parsed = urlparse(url)
        if any(domain in parsed.netloc for domain in shortened_domains):
            print(f"🔗 Resolving shortened URL: {url}")
            response = requests.head(url, allow_redirects=True, timeout=timeout)
            resolved = response.url
            print(f"✅ Resolved to: {resolved}")
            return resolved
    except Exception as e:
        print(f"⚠️ Could not resolve shortened URL {url}: {e}")
    return url

def get_sheet_data(force_refresh=False):
    if not force_refresh and is_cache_valid():
        cache_age = (datetime.now() - cache_data['timestamp']).seconds
        print(f"✅ Using cached data (age: {cache_age}s)")
        return cache_data['data']
    try:
        print("📡 Fetching fresh data from Google Sheets...")
        credentials = get_google_credentials()
        client = gspread.authorize(credentials)
        sheet = client.open_by_url(SHEET_URL).worksheet(WORKSHEET_NAME)
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        print(f"✅ Successfully loaded {len(df)} records from '{WORKSHEET_NAME}'")
        numeric_columns = ['TRUST SCORE RATING', 'total_revenue', 'total_order_action', 'total_clicks', 
                          'Estimated Avg. Ahrefs DR', 'Estimated Avg. Semrush AS', 'Estimated Avg. Moz DA', 'CR']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('[\$,%]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        text_cols = ['Clean Up Name', 'Description', 'Industry Vertical', 'Company Business Model', 
                     'Sub Company Vertical', 'Website', 'Associated Contact', 'RBL Brand', 'Affiliate_Brand']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        print("🔧 Pre-filtering quality publishers...")
        df_filtered = filter_quality_publishers(df)
        print(f"✅ Filtered to {len(df_filtered)} quality publishers")
        print("📝 Building weighted content for TF-IDF...")
        df_filtered['weighted_content'] = df_filtered.apply(create_weighted_content, axis=1)
        df_filtered['partner_normalized'] = df_filtered['Clean Up Name'].str.strip().str.lower()
        print("🤖 Building TF-IDF vectorizer and matrix (this runs ONCE per cache refresh)...")
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=2500,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.90
            )
            tfidf_matrix = vectorizer.fit_transform(df_filtered['weighted_content'].fillna(''))
            cache_data['tfidf_vectorizer'] = vectorizer
            cache_data['tfidf_matrix'] = tfidf_matrix
            cache_data['filtered_df'] = df_filtered
            print(f"✅ TF-IDF matrix built and cached: {tfidf_matrix.shape}")
            print(f"   This will make all searches INSTANT for the next 30 minutes!")
        except Exception as e:
            print(f"⚠️ Error building TF-IDF matrix: {e}")
            cache_data['tfidf_vectorizer'] = None
            cache_data['tfidf_matrix'] = None
            cache_data['filtered_df'] = df_filtered
        cache_data['data'] = df
        cache_data['timestamp'] = datetime.now()
        cache_data['partner_names'] = df['Clean Up Name'].dropna().astype(str).unique().tolist()
        print(f"🔄 Cache updated at {cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        return df
    except Exception as e:
        print(f"❌ Error loading Google Sheets data: {e}")
        if cache_data['data'] is not None:
            print("⚠️ Returning expired cache data due to error")
            return cache_data['data']
        return pd.DataFrame()

def filter_quality_publishers(df):
    if df.empty:
        return df
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    df_filtered = df_filtered[
        df_filtered['Clean Up Name'].notna() & 
        (df_filtered['Clean Up Name'].str.strip() != '')
    ]
    df_filtered = df_filtered[
        (df_filtered['Clean Up Name'].str.len() <= 75) &
        (df_filtered['Clean Up Name'].str.contains(' ', na=False) | 
         (df_filtered['Clean Up Name'].str.len() <= 20))
    ]
    required_fields = {
        'TRUST SCORE RATING': lambda x: x > 0,
        'Estimated Avg. Ahrefs DR': lambda x: x > 0,
        'Estimated Avg. Moz DA': lambda x: x > 0,
        'Estimated Avg. Semrush AS': lambda x: x > 0,
        'Website': lambda x: pd.notna(x) and str(x).strip() != '',
        'Associated Contact': lambda x: pd.notna(x) and str(x).strip() != '',
        'Description': lambda x: pd.notna(x) and str(x).strip() != '',
        'Industry Vertical': lambda x: pd.notna(x) and str(x).strip() != '',
        'Company Business Model': lambda x: pd.notna(x) and str(x).strip() != '',
        'Sub Company Vertical': lambda x: pd.notna(x) and str(x).strip() != ''
    }
    for field, validator in required_fields.items():
        if field in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[field].apply(validator)]
    invalid_domains = ['t.co', 'drive.google.com', 'webflow.com', 'canva.com', 
                      'prnt.sc', 'bit.ly', 'bitly.com', 'tinyurl.com', 'goo.gl',
                      'docs.google.com', 'forms.google.com', 'sheets.google.com',
                      'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
                      'linkedin.com', 'youtube.com', 'ow.ly', 'buff.ly']
    def is_valid_website(url):
        if pd.isna(url) or not url.strip():
            return False
        url_lower = url.lower().strip()
        return not any(invalid_domain in url_lower for invalid_domain in invalid_domains)
    df_filtered = df_filtered[df_filtered['Website'].apply(is_valid_website)]
    df_filtered = df_filtered[
        ~df_filtered['Clean Up Name'].str.lower().str.contains('client', na=False, regex=False)
    ]
    df_filtered = df_filtered[df_filtered['TRUST SCORE RATING'] >= 20]
    removed_count = initial_count - len(df_filtered)
    print(f"📊 Quality filter: {initial_count} → {len(df_filtered)} publishers (removed {removed_count})")
    return df_filtered

def create_weighted_content(row):
    content_parts = []
    industry = str(row.get('Industry Vertical', '')).strip()
    if industry:
        content_parts.extend([industry] * 10)
    sub_vertical = str(row.get('Sub Company Vertical', '')).strip()
    if sub_vertical:
        content_parts.extend([sub_vertical] * 8)
    description = str(row.get('Description', '')).strip()
    if description:
        content_parts.extend([description] * 7)
    business_model = str(row.get('Company Business Model', '')).strip()
    if business_model:
        content_parts.extend([business_model] * 6)
    return ' '.join(content_parts)

def extract_enhanced_content_from_url(url):
    try:
        url = resolve_shortened_url(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()[:1000]
        if not is_english_text(page_text):
            print(f"⚠️ Non-English page detected, skipping: {url}")
            return ""
        content_parts = []
        description = soup.find('meta', {'name': 'description'})
        if description and description.get('content', '').strip():
            desc_text = description.get('content').strip()
            content_parts.extend([desc_text] * 8)
            print(f"📝 Meta Description: {desc_text[:150]}")
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            content_parts.extend([title_tag.get_text().strip()] * 6)
            print(f"📄 Title: {title_tag.get_text().strip()[:100]}")
        for h1 in soup.find_all('h1')[:5]:
            heading_text = h1.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 4)
        for h2 in soup.find_all('h2')[:20]:
            heading_text = h2.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 3)
        final_content = ' '.join(content_parts)
        print(f"📄 Extracted {len(final_content)} characters from URL")
        return final_content
    except Exception as e:
        print(f"❌ Error extracting content from URL: {e}")
        return ""

def calculate_combined_score(similarity_scores, trust_scores, has_rbl_brand, similarity_weight=0.6, trust_weight=0.2, rbl_weight=0.2):
    max_trust_score = 100
    normalized_trust = np.array(trust_scores) / max_trust_score
    normalized_similarity = np.array(similarity_scores)
    rbl_bonus = np.array(has_rbl_brand, dtype=float)
    combined_scores = (
        (similarity_weight * normalized_similarity) + 
        (trust_weight * normalized_trust) + 
        (rbl_weight * rbl_bonus)
    )
    return combined_scores

def prioritize_rbl_publishers(publishers, min_rbl_count=20):
    if not publishers:
        return publishers
    rbl_publishers = [p for p in publishers if p.get('RBL Brand', '').strip()]
    non_rbl_publishers = [p for p in publishers if not p.get('RBL Brand', '').strip()]
    print(f"📊 RBL publishers found: {len(rbl_publishers)}, Non-RBL: {len(non_rbl_publishers)}")
    result = []
    rbl_idx = 0
    non_rbl_idx = 0
    while len(result) < len(publishers):
        if rbl_idx < len(rbl_publishers):
            result.append(rbl_publishers[rbl_idx])
            rbl_idx += 1
        if non_rbl_idx < len(non_rbl_publishers) and len(result) < len(publishers):
            result.append(non_rbl_publishers[non_rbl_idx])
            non_rbl_idx += 1
    print(f"✅ Final mix: {len([p for p in result if p.get('RBL Brand', '').strip()])} RBL, {len([p for p in result if not p.get('RBL Brand', '').strip()])} non-RBL")
    return result

def deduplicate_publishers(publishers):
    if not publishers:
        return publishers
    url_groups = {}
    for pub in publishers:
        website = str(pub.get('Website', '')).strip().lower()
        website = website.replace('www.', '').replace('http://', '').replace('https://', '').rstrip('/')
        if not website:
            continue
        if website not in url_groups:
            url_groups[website] = []
        url_groups[website].append(pub)
    deduplicated = []
    removed_count = 0
    for website, pubs in url_groups.items():
        if len(pubs) == 1:
            deduplicated.append(pubs[0])
        else:
            best_pub = max(pubs, key=lambda p: (
                bool(p.get('RBL Brand', '').strip()),
                len(str(p.get('Clean Up Name', ''))),
                p.get('TRUST SCORE RATING', 0)
            ))
            deduplicated.append(best_pub)
            removed_count += len(pubs) - 1
    if removed_count > 0:
        print(f"🔄 Deduplication removed {removed_count} duplicates")
    return deduplicated

# -------------------------------
# Routes - OPTIMIZED
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    try:
        query = request.args.get('query', '').strip().lower()
        if len(query) < 2:
            return jsonify([])
        if cache_data['partner_names'] is not None:
            partner_names = cache_data['partner_names']
        else:
            df = get_sheet_data()
            if df.empty:
                return jsonify([])
            partner_names = df['Clean Up Name'].dropna().astype(str).unique().tolist()
        suggestions = [name for name in partner_names if query in name.lower()]
        return jsonify(sorted(suggestions[:20]))
    except Exception as e:
        print(f"❌ Error in autocomplete: {e}")
        return jsonify([])

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    try:
        cache_data['data'] = None
        cache_data['timestamp'] = None
        cache_data['partner_names'] = None
        cache_data['tfidf_vectorizer'] = None
        cache_data['tfidf_matrix'] = None
        cache_data['filtered_df'] = None
        print("🔄 Cache cleared manually, fetching fresh data...")
        df = get_sheet_data(force_refresh=True)
        if df.empty:
            return jsonify({"success": False, "message": "Failed to load data from Google Sheets"}), 500
        return jsonify({
            "success": True, 
            "message": f"Data refreshed successfully! Loaded {len(df)} records.",
            "record_count": len(df),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"❌ Error refreshing data: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/cache_status')
def cache_status():
    try:
        if cache_data['timestamp'] is None:
            return jsonify({
                "cached": False,
                "message": "No data in cache"
            })
        cache_age = (datetime.now() - cache_data['timestamp']).seconds
        is_valid = is_cache_valid()
        return jsonify({
            "cached": True,
            "valid": is_valid,
            "age_seconds": cache_age,
            "age_minutes": round(cache_age / 60, 1),
            "last_updated": cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "record_count": len(cache_data['data']) if cache_data['data'] is not None else 0,
            "cache_duration_minutes": CACHE_DURATION.seconds // 60
        })
    except Exception as e:
        print(f"❌ Error getting cache status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/result', methods=['POST'])
def result():
    try:
        partner_input = request.form.get('partner_name', '').strip()
        print(f"🔍 User searched for: {partner_input}")
        df = get_sheet_data()
        if df.empty:
            return render_template('result_trust_score.html', error="Unable to load data from Google Sheets.")
        df['partner_normalized'] = df['Clean Up Name'].str.strip().str.lower()
        partner_normalized = partner_input.lower()
        if partner_normalized not in df['partner_normalized'].values:
            return render_template('result_trust_score.html', error=f"Partner '{partner_input}' not found.")
        selected = df[df['partner_normalized'] == partner_normalized]
        if selected.empty:
            return render_template('result_trust_score.html', error="No data found for the selected partner.")
        trust_score = selected['TRUST SCORE RATING'].mean()
        partner_display_name = selected['Clean Up Name'].iloc[0]
        affiliate_brands = (
            selected.groupby('Affiliate_Brand', as_index=False)
            .agg({
                'total_revenue': 'sum',
                'total_order_action': 'sum',
                'total_clicks': 'sum',
                'CR': 'mean'
            })
        ).to_dict('records')
        selected_content = create_weighted_content(selected.iloc[0])
        if selected_content.strip() and cache_data['tfidf_vectorizer'] is not None:
            try:
                print("⚡ Using cached TF-IDF vectorizer (INSTANT!)")
                vectorizer = cache_data['tfidf_vectorizer']
                tfidf_matrix = cache_data['tfidf_matrix']
                df_filtered = cache_data['filtered_df'].copy()
                search_vector = vectorizer.transform([selected_content])
                similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
                df_filtered['similarity'] = similarity_scores
                df_filtered = df_filtered[df_filtered['partner_normalized'] != partner_normalized]
                df_filtered['has_rbl_brand'] = df_filtered['RBL Brand'].apply(lambda x: bool(str(x).strip()))
                combined_scores = calculate_combined_score(
                    df_filtered['similarity'].values,
                    df_filtered['TRUST SCORE RATING'].values,
                    df_filtered['has_rbl_brand'].values
                )
                df_filtered['combined_score'] = combined_scores
                similar_df = df_filtered.sort_values(
                    by='combined_score',
                    ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
            except Exception as e:
                print(f"⚠️ TF-IDF failed, using trust score fallback: {e}")
                df_filtered = filter_quality_publishers(df)
                df_filtered['partner_normalized'] = df_filtered['Clean Up Name'].str.strip().str.lower()
                similar_df = df_filtered[df_filtered['partner_normalized'] != partner_normalized].sort_values(
                    by='TRUST SCORE RATING', ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
        else:
            df_filtered = filter_quality_publishers(df)
            df_filtered['partner_normalized'] = df_filtered['Clean Up Name'].str.strip().str.lower()
            similar_df = df_filtered[df_filtered['partner_normalized'] != partner_normalized].sort_values(
                by='TRUST SCORE RATING', ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(100)
        similar_partners = similar_df[[
            'Clean Up Name', 'TRUST SCORE RATING', 'Estimated Avg. Ahrefs DR',
            'Estimated Avg. Moz DA', 'Estimated Avg. Semrush AS', 'Website',
            'Associated Contact', 'Description', 'Industry Vertical',
            'Company Business Model', 'Sub Company Vertical', 'RBL Brand'
        ]].to_dict('records')
        similar_partners = deduplicate_publishers(similar_partners)
        similar_partners = prioritize_rbl_publishers(similar_partners, min_rbl_count=20)
        similar_partners = similar_partners[:40]
        return render_template(
            'result_trust_score.html',
            partner_name=partner_display_name,
            trust_score=trust_score,
            affiliate_brands=affiliate_brands,
            similar_partners=similar_partners
        )
    except Exception as e:
        print(f"❌ Error in /result: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result_trust_score.html', error="An unexpected error occurred.")

@app.route('/publisher_lookup', methods=['POST'])
def publisher_lookup():
    try:
        brand_url = request.form.get('brand_url', '').strip()
        print(f"🔍 Brand URL lookup: {brand_url}")
        url_content = extract_enhanced_content_from_url(brand_url)
        if not url_content:
            return render_template('result_brand_lookup.html', 
                                 error="Unable to extract content from the URL.",
                                 brand_url=brand_url)
        df = get_sheet_data()
        if df.empty:
            return render_template('result_brand_lookup.html', 
                                 error="Unable to load data from Google Sheets.",
                                 brand_url=brand_url)
        if cache_data['tfidf_vectorizer'] is not None:
            try:
                print("⚡ Using cached TF-IDF vectorizer (INSTANT!)")
                vectorizer = cache_data['tfidf_vectorizer']
                tfidf_matrix = cache_data['tfidf_matrix']
                df_filtered = cache_data['filtered_df'].copy()
                search_vector = vectorizer.transform([url_content])
                similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
                df_filtered['similarity'] = similarity_scores
                min_similarity_threshold = 0.12
                df_filtered = df_filtered[df_filtered['similarity'] >= min_similarity_threshold]
                print(f"📊 After similarity filter: {len(df_filtered)} publishers with similarity >= {min_similarity_threshold}")
                if df_filtered.empty:
                    return render_template('result_brand_lookup.html', 
                        error="No relevant publishers found for this brand.",
                        brand_url=brand_url)
                df_filtered['has_rbl_brand'] = df_filtered['RBL Brand'].apply(lambda x: bool(str(x).strip()))
                combined_scores = calculate_combined_score(
                    df_filtered['similarity'].values,
                    df_filtered['TRUST SCORE RATING'].values,
                    df_filtered['has_rbl_brand'].values
                )
                df_filtered['combined_score'] = combined_scores
                publishers_df = df_filtered.sort_values(
                    by='combined_score',
                    ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
            except Exception as e:
                print(f"⚠️ TF-IDF failed in brand lookup, using trust score: {e}")
                df_filtered = filter_quality_publishers(df)
                publishers_df = df_filtered.sort_values(
                    by='TRUST SCORE RATING', ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
        else:
            df_filtered = filter_quality_publishers(df)
            publishers_df = df_filtered.sort_values(
                by='TRUST SCORE RATING', ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(100)
        publishers = publishers_df[[
            'Clean Up Name', 'TRUST SCORE RATING', 'Estimated Avg. Ahrefs DR',
            'Estimated Avg. Moz DA', 'Estimated Avg. Semrush AS', 'Website',
            'Associated Contact', 'Description', 'Industry Vertical',
            'Company Business Model', 'Sub Company Vertical', 'RBL Brand'
        ]].to_dict('records')
        publishers = deduplicate_publishers(publishers)
        publishers = prioritize_rbl_publishers(publishers, min_rbl_count=20)
        publishers = publishers[:40]
        return render_template('result_brand_lookup.html', 
                             publishers=publishers,
                             brand_url=brand_url)
    except Exception as e:
        print(f"❌ Error in /publisher_lookup: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result_brand_lookup.html', 
                             error="An unexpected error occurred.",
                             brand_url=brand_url if 'brand_url' in locals() else None)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        data = request.json.get('publishers', [])
        if not data:
            return jsonify({"error": "No data provided"}), 400
        df_export = pd.DataFrame(data)
        column_mapping = {
            'Clean Up Name': 'Partner Name',
            'TRUST SCORE RATING': 'Trust Partner Score',
            'Estimated Avg. Ahrefs DR': 'Avg. Ahrefs',
            'Estimated Avg. Moz DA': 'Avg. Moz',
            'Estimated Avg. Semrush AS': 'Avg. Semrush',
            'Website': 'URL',
            'Associated Contact': 'Contact',
            'Description': 'Description',
            'Industry Vertical': 'Industry Vertical',
            'Company Business Model': 'Business Model',
            'Sub Company Vertical': 'Company Vertical',
            'RBL Brand': 'RBL Brand'
        }
        df_export = df_export.rename(columns=column_mapping)
        desired_order = [
            'Partner Name', 'Trust Partner Score', 'Avg. Ahrefs', 'Avg. Moz', 
            'Avg. Semrush', 'URL', 'Contact', 'Description', 
            'Industry Vertical', 'Business Model', 'Company Vertical', 'RBL Brand'
        ]
        existing_cols = [col for col in desired_order if col in df_export.columns]
        df_export = df_export[existing_cols]
        output = io.StringIO()
        df_export.to_csv(output, index=False)
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=similar_partners.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")
        return jsonify({"error": "Failed to generate CSV"}), 500

# -------------------------------
# Run Application
# -------------------------------
if __name__ == '__main__':
    print("🔥 Pre-warming cache on startup...")
    try:
        get_sheet_data(force_refresh=True)
        print("✅ Cache pre-warmed successfully! App is ready for INSTANT responses!")
    except Exception as e:
        print(f"⚠️ Failed to pre-warm cache: {e}")
        print("⚠️ App will still work but first request will be slow")
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)