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

# Cache configuration
CACHE_DURATION = timedelta(minutes=30)
cache_data = {
    'data': None,
    'timestamp': None,
    'partner_names': None
}

# -------------------------------
# Helper Functions
# -------------------------------

def get_google_credentials():
    """Get Google credentials from environment variable or file"""
    try:
        # Try to get credentials from environment variable first (for Render)
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        
        if creds_json:
            print("✅ Using credentials from environment variable")
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            )
        else:
            # Fall back to credentials.json file (for local development)
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
    """Check if cache is still valid"""
    if cache_data['data'] is None or cache_data['timestamp'] is None:
        return False
    
    time_elapsed = datetime.now() - cache_data['timestamp']
    return time_elapsed < CACHE_DURATION

def is_english_text(text):
    """
    Detect if text is primarily in English
    Returns True if English, False otherwise
    """
    if not text or len(text.strip()) < 10:
        return False
    
    try:
        detected_lang = detect(text)
        is_eng = detected_lang == 'en'
        if not is_eng:
            print(f"🌐 Non-English text detected: {detected_lang}")
        return is_eng
    except LangDetectException:
        # If detection fails, check for Latin characters
        latin_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text.strip())
        return total_chars > 0 and (latin_chars / total_chars) >= 0.7

def resolve_shortened_url(url, timeout=10):
    """
    Resolve shortened URLs (t.co, bit.ly, etc.) to their final destination
    """
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
    """Load data from Google Sheets with caching (30 min) and optional force refresh"""
    # Check if cache is valid and force_refresh is False
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
        
        print(f"✅ Successfully loaded sheet data from '{WORKSHEET_NAME}'")
        print(f"📊 Loaded {len(df)} records")
        
        # Clean numeric columns
        numeric_columns = ['TRUST SCORE RATING', 'total_revenue', 'total_order_action', 'total_clicks', 
                          'Estimated Avg. Ahrefs DR', 'Estimated Avg. Semrush AS', 'Estimated Avg. Moz DA', 'CR']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('[\$,%]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill NaN values for text columns
        text_cols = ['Clean Up Name', 'Description', 'Industry Vertical', 'Company Business Model', 
                     'Sub Company Vertical', 'Website', 'Associated Contact', 'RBL Brand', 'Affiliate_Brand']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Update cache
        cache_data['data'] = df
        cache_data['timestamp'] = datetime.now()
        
        # Cache partner names for autocomplete
        cache_data['partner_names'] = df['Clean Up Name'].dropna().astype(str).unique().tolist()
        
        if force_refresh:
            print(f"🔄 Cache manually refreshed at {cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"🔄 Cache updated at {cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading Google Sheets data: {e}")
        # Return cached data if available, even if expired
        if cache_data['data'] is not None:
            print("⚠️ Returning expired cache data due to error")
            return cache_data['data']
        return pd.DataFrame()

def filter_quality_publishers(df):
    """
    Filter out publishers with poor data quality before matching.
    Returns only publishers with valid, high-quality data.
    ENHANCED with language detection and better validation.
    """
    if df.empty:
        return df
    
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    
    # 1. Filter out rows with missing or invalid partner names
    df_filtered = df_filtered[
        df_filtered['Clean Up Name'].notna() & 
        (df_filtered['Clean Up Name'].str.strip() != '')
    ]
    
    # 2. Filter out partner names that are too long (likely corrupted data)
    # Also filter names with no spaces (likely mangled)
    df_filtered = df_filtered[
        (df_filtered['Clean Up Name'].str.len() <= 75) &
        (df_filtered['Clean Up Name'].str.contains(' ', na=False) | 
         (df_filtered['Clean Up Name'].str.len() <= 20))  # Allow short single-word names
    ]
    
    # 3. REQUIRED FIELDS - Must have ALL of these fields with valid data
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
    
    # Apply required field filters
    for field, validator in required_fields.items():
        if field in df_filtered.columns:
            before_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered[field].apply(validator)]
            removed = before_count - len(df_filtered)
            if removed > 0:
                print(f"   🔍 Removed {removed} entries with invalid '{field}'")
    
    # 4. Filter out invalid domain names (common invalid patterns)
    invalid_domains = [
        't.co', 'drive.google.com', 'webflow.com', 'canva.com', 
        'prnt.sc', 'bit.ly', 'bitly.com', 'tinyurl.com', 'goo.gl',
        'docs.google.com', 'forms.google.com', 'sheets.google.com',
        'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'ow.ly', 'buff.ly'
    ]
    
    def is_valid_website(url):
        if pd.isna(url) or not url.strip():
            return False
        url_lower = url.lower().strip()
        # Check if URL contains any invalid domains
        is_invalid = any(invalid_domain in url_lower for invalid_domain in invalid_domains)
        if is_invalid:
            print(f"   🚫 Invalid domain filtered: {url_lower[:50]}")
        return not is_invalid
    
    before_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['Website'].apply(is_valid_website)]
    removed = before_count - len(df_filtered)
    if removed > 0:
        print(f"   🌐 Removed {removed} entries with invalid/shortened URLs")
    
    # 5. Filter out specific client names (exact matches, case-insensitive)
    excluded_clients = [
        'Somnee - Client',
        'SCJ - Client',
        'Thermacell - Client',
        'Remote - Client',
        'Reflux Gourmet - Client',
        'Grammarly - Client',
        'Future - Client',
        'FinanceHQ LLC - Client',
        'Tiktok - Client',
        'Atlassian - Client'
    ]
    
    # Create a normalized version for comparison
    excluded_clients_lower = [name.lower().strip() for name in excluded_clients]
    df_filtered = df_filtered[
        ~df_filtered['Clean Up Name'].str.lower().str.strip().isin(excluded_clients_lower)
    ]
    
    # 6. Filter out rows where partner name contains client/brand keywords
    client_keywords = ['client', ' client$', 'affiliate client']
    pattern = '|'.join(client_keywords)
    before_count = len(df_filtered)
    df_filtered = df_filtered[
        ~df_filtered['Clean Up Name'].str.lower().str.contains(pattern, na=False, regex=True)
    ]
    removed = before_count - len(df_filtered)
    if removed > 0:
        print(f"   👥 Removed {removed} entries with 'client' keywords")
    
    # 7. Filter out descriptions with non-English text - ENHANCED
    def has_valid_description(desc):
        if pd.isna(desc) or not desc.strip():
            return False
        
        # First check: Must have mostly Latin characters
        latin_chars = sum(1 for c in desc if ord(c) < 128)
        total_chars = len(desc.strip())
        if total_chars == 0 or (latin_chars / total_chars) < 0.7:
            return False
        
        # Second check: Use language detection for longer descriptions
        if len(desc.strip()) > 50:
            return is_english_text(desc)
        
        return True
    
    before_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['Description'].apply(has_valid_description)]
    removed = before_count - len(df_filtered)
    if removed > 0:
        print(f"   🌍 Removed {removed} entries with non-English descriptions")
    
    # 8. Require minimum trust score
    min_trust_score = 20
    before_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['TRUST SCORE RATING'] >= min_trust_score]
    removed = before_count - len(df_filtered)
    if removed > 0:
        print(f"   ⭐ Removed {removed} entries with trust score < {min_trust_score}")
    
    removed_count = initial_count - len(df_filtered)
    print(f"📊 Quality filter: {initial_count} → {len(df_filtered)} publishers (removed {removed_count})")
    
    return df_filtered

def create_weighted_content(row):
    """
    Create weighted content string for TF-IDF with prioritized fields
    
    Priority Order (weights):
    1. Industry Vertical: 10x (HIGHEST - primary matching factor)
    2. Sub Company Vertical: 8x (VERY HIGH)
    3. Description: 7x (HIGH - must align with meta tags/content)
    4. Company Business Model: 6x (HIGH)
    """
    content_parts = []
    
    # HIGHEST Priority: Industry Vertical (weight: 10x)
    industry = str(row.get('Industry Vertical', '')).strip()
    if industry:
        content_parts.extend([industry] * 10)
    
    # VERY HIGH Priority: Sub Company Vertical (weight: 8x)
    sub_vertical = str(row.get('Sub Company Vertical', '')).strip()
    if sub_vertical:
        content_parts.extend([sub_vertical] * 8)
    
    # HIGH Priority: Description (weight: 7x)
    description = str(row.get('Description', '')).strip()
    if description:
        content_parts.extend([description] * 7)
    
    # HIGH Priority: Company Business Model (weight: 6x)
    business_model = str(row.get('Company Business Model', '')).strip()
    if business_model:
        content_parts.extend([business_model] * 6)
    
    return ' '.join(content_parts)

def extract_enhanced_content_from_url(url):
    """
    Enhanced content extraction from URL with URL resolution and language detection
    
    Priority Order (weights):
    1. Category/Industry from structured data: 10x
    2. Meta description: 8x
    3. Meta keywords: 7x
    4. Title tags: 6x
    5. Main content/features: 5x
    6. Headings: 4x
    7. Navigation: 3x
    8. Brand name: 2x
    """
    try:
        # Resolve shortened URLs first
        original_url = url
        url = resolve_shortened_url(url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check if page is in English (sample first 1000 characters)
        page_text = soup.get_text()[:1000]
        if not is_english_text(page_text):
            print(f"⚠️ Non-English page detected, skipping: {url}")
            return ""
        
        content_parts = []
        
        # 1. HIGHEST: Extract CATEGORY/INDUSTRY keywords from structured data (weight: 10x)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts[:5]:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    category = data.get('applicationCategory', '') or data.get('category', '')
                    industry_type = data.get('industry', '') or data.get('@type', '')
                    org_type = data.get('organizationType', '')
                    
                    if category:
                        content_parts.extend([category] * 10)
                        print(f"🏭 Category found: {category}")
                    if industry_type and industry_type not in ['Organization', 'Corporation', 'WebSite']:
                        content_parts.extend([industry_type] * 10)
                        print(f"🏭 Industry type: {industry_type}")
                    if org_type:
                        content_parts.extend([org_type] * 8)
            except:
                pass
        
        # 2. VERY HIGH: Meta description (weight: 8x)
        description = soup.find('meta', {'name': 'description'})
        if description and description.get('content', '').strip():
            desc_text = description.get('content').strip()
            content_parts.extend([desc_text] * 8)
            print(f"📝 Meta Description: {desc_text[:150]}")
        
        og_description = soup.find('meta', property='og:description')
        if og_description and og_description.get('content', '').strip():
            content_parts.extend([og_description.get('content').strip()] * 7)
        
        twitter_description = soup.find('meta', {'name': 'twitter:description'})
        if twitter_description and twitter_description.get('content', '').strip():
            content_parts.extend([twitter_description.get('content').strip()] * 7)
        
        # 3. HIGH: Meta keywords (weight: 7x)
        keywords = soup.find('meta', {'name': 'keywords'})
        if keywords and keywords.get('content', '').strip():
            keywords_text = keywords.get('content').strip()
            content_parts.extend([keywords_text] * 7)
            print(f"🔑 Keywords: {keywords_text[:100]}")
        
        # 4. HIGH: Title tags (weight: 6x)
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            content_parts.extend([title_tag.get_text().strip()] * 6)
            print(f"📄 Title: {title_tag.get_text().strip()[:100]}")
        
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content', '').strip():
            content_parts.extend([og_title.get('content').strip()] * 5)
        
        # 5. MEDIUM-HIGH: Main content (weight: 5x)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'role': 'main'})
        if main_content:
            feature_sections = main_content.find_all(['ul', 'ol'])
            for section in feature_sections[:10]:
                section_text = section.get_text().strip()
                if len(section_text) > 30:
                    content_parts.extend([section_text] * 5)
            
            paragraphs = main_content.find_all('p')
            for p in paragraphs[:25]:
                text = p.get_text().strip()
                if len(text) > 50:
                    content_parts.extend([text] * 3)
        
        # 6. MEDIUM: Headings (weight: 4x for H1, 3x for H2)
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags[:5]:
            heading_text = h1.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 4)
        
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags[:20]:
            heading_text = h2.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 3)
        
        # 7. MEDIUM: Navigation (weight: 3x)
        nav_elements = soup.find_all(['nav', 'header'])
        for nav in nav_elements[:5]:
            nav_text = ' '.join([a.get_text().strip() for a in nav.find_all('a') if a.get_text().strip()])
            if nav_text:
                content_parts.extend([nav_text] * 3)
        
        # 8. LOW: Brand name (weight: 2x)
        brand_selectors = [
            soup.find('meta', property='og:site_name'),
            soup.find('meta', {'name': 'application-name'}),
        ]
        
        for selector in brand_selectors:
            if selector:
                brand = selector.get('content', '') if selector.get('content') else selector.get_text()
                if brand.strip():
                    content_parts.extend([brand.strip()] * 2)
                    print(f"🏢 Brand: {brand.strip()}")
                    break
        
        final_content = ' '.join(content_parts)
        print(f"📄 Extracted {len(final_content)} characters from URL")
        return final_content
        
    except Exception as e:
        print(f"❌ Error extracting content from URL: {e}")
        return ""

def calculate_combined_score(similarity_scores, trust_scores, has_rbl_brand, similarity_weight=0.6, trust_weight=0.2, rbl_weight=0.2):
    """Combine similarity scores, trust scores, and RBL brand preference"""
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
    """Ensure at least min_rbl_count publishers with RBL Brand are in the results"""
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
    """
    ENHANCED: Deduplicate publishers based on website URL (primary key)
    Keep the entry with the longest/most complete name and highest trust score
    """
    if not publishers:
        return publishers
    
    # Group by normalized website URL
    url_groups = {}
    
    for pub in publishers:
        website = str(pub.get('Website', '')).strip().lower()
        # Normalize URL
        website = website.replace('www.', '').replace('http://', '').replace('https://', '').rstrip('/')
        
        if not website:
            continue
        
        if website not in url_groups:
            url_groups[website] = []
        url_groups[website].append(pub)
    
    # For each URL group, keep the best entry
    deduplicated = []
    removed_count = 0
    
    for website, pubs in url_groups.items():
        if len(pubs) == 1:
            deduplicated.append(pubs[0])
        else:
            # Multiple entries for same website - pick the best one
            # Priority: 1) Has RBL Brand, 2) Longest name, 3) Highest trust score
            best_pub = max(pubs, key=lambda p: (
                bool(p.get('RBL Brand', '').strip()),  # RBL first
                len(str(p.get('Clean Up Name', ''))),   # Then longest name
                p.get('TRUST SCORE RATING', 0)          # Then highest trust score
            ))
            
            deduplicated.append(best_pub)
            removed_count += len(pubs) - 1
            
            # Log what was removed
            removed_names = [p.get('Clean Up Name', '') for p in pubs if p != best_pub]
            kept_name = best_pub.get('Clean Up Name', '')
            print(f"   🔄 Kept '{kept_name}', removed: {', '.join(removed_names)}")
    
    if removed_count > 0:
        print(f"🔄 Deduplication removed {removed_count} duplicates")
    
    return deduplicated

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def index():
    """Homepage with search forms"""
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    """Autocomplete endpoint for partner names"""
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
    """Manually refresh data from Google Sheets"""
    try:
        cache_data['data'] = None
        cache_data['timestamp'] = None
        cache_data['partner_names'] = None
        
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
    """Get current cache status"""
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
    """Trust Score Lookup Result Page"""
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
        
        df_filtered = filter_quality_publishers(df)
        df_filtered['weighted_content'] = df_filtered.apply(create_weighted_content, axis=1)
        selected_content = create_weighted_content(selected.iloc[0])
        
        if selected_content.strip() and not df_filtered.empty:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    max_features=2500,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.90
                )
                all_content = df_filtered['weighted_content'].fillna('').tolist() + [selected_content]
                tfidf_matrix = vectorizer.fit_transform(all_content)
                similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                
                df_copy = df_filtered.copy()
                df_copy['similarity'] = similarity_scores
                df_copy = df_copy[df_copy['partner_normalized'] != partner_normalized]
                df_copy['has_rbl_brand'] = df_copy['RBL Brand'].apply(lambda x: bool(str(x).strip()))
                
                combined_scores = calculate_combined_score(
                    df_copy['similarity'].values,
                    df_copy['TRUST SCORE RATING'].values,
                    df_copy['has_rbl_brand'].values
                )
                df_copy['combined_score'] = combined_scores
                
                similar_df = df_copy.sort_values(
                    by='combined_score',
                    ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
            
            except Exception as e:
                print(f"⚠️ TF-IDF failed, using trust score fallback: {e}")
                similar_df = df_filtered[df_filtered['partner_normalized'] != partner_normalized].sort_values(
                    by='TRUST SCORE RATING', ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(100)
        else:
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
        return render_template('result_trust_score.html', error="An unexpected error occurred.")

@app.route('/publisher_lookup', methods=['POST'])
def publisher_lookup():
    """Brand URL Lookup Result Page"""
    try:
        brand_url = request.form.get('brand_url', '').strip()
        print(f"🔍 Brand URL lookup: {brand_url}")
        
        url_content = extract_enhanced_content_from_url(brand_url)
        if not url_content:
            return render_template('result_brand_lookup.html', error="Unable to extract content from the URL.")
        
        df = get_sheet_data()
        if df.empty:
            return render_template('result_brand_lookup.html', error="Unable to load data from Google Sheets.")
        
        df = filter_quality_publishers(df)
        
        if df.empty:
            return render_template('result_brand_lookup.html', error="No high-quality publishers found in the database.")
        
        df['weighted_content'] = df.apply(create_weighted_content, axis=1)
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=2500,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.90
            )
            all_content = df['weighted_content'].fillna('').tolist() + [url_content]
            tfidf_matrix = vectorizer.fit_transform(all_content)
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            df_copy = df.copy()
            df_copy['similarity'] = similarity_scores
            
            min_similarity_threshold = 0.12
            df_copy = df_copy[df_copy['similarity'] >= min_similarity_threshold]
            
            print(f"📊 After similarity filter: {len(df_copy)} publishers with similarity >= {min_similarity_threshold}")
            
            if df_copy.empty:
                return render_template('result_brand_lookup.html', 
                    error="No relevant publishers found for this brand.")
            
            df_copy['has_rbl_brand'] = df_copy['RBL Brand'].apply(lambda x: bool(str(x).strip()))
            
            combined_scores = calculate_combined_score(
                df_copy['similarity'].values,
                df_copy['TRUST SCORE RATING'].values,
                df_copy['has_rbl_brand'].values
            )
            df_copy['combined_score'] = combined_scores
            
            publishers_df = df_copy.sort_values(
                by='combined_score',
                ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(100)
        
        except Exception as e:
            print(f"⚠️ TF-IDF failed in brand lookup, using trust score: {e}")
            publishers_df = df.sort_values(
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
        
        return render_template('result_brand_lookup.html', publishers=publishers)
    
    except Exception as e:
        print(f"❌ Error in /publisher_lookup: {e}")
        return render_template('result_brand_lookup.html', error="An unexpected error occurred.")

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download partners data as CSV"""
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
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)