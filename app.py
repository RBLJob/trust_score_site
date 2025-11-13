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

def get_sheet_data():
    """Load data from Google Sheets with caching (30 min)"""
    # Check if cache is valid
    if is_cache_valid():
        print(f"✅ Using cached data (age: {(datetime.now() - cache_data['timestamp']).seconds}s)")
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
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading Google Sheets data: {e}")
        # Return cached data if available, even if expired
        if cache_data['data'] is not None:
            print("⚠️ Returning expired cache data due to error")
            return cache_data['data']
        return pd.DataFrame()

def create_weighted_content(row):
    """Create weighted content string for TF-IDF with prioritized fields"""
    content_parts = []
    
    # Highest Priority: Industry Vertical (weight: 4x)
    industry = str(row.get('Industry Vertical', '')).strip()
    if industry:
        content_parts.extend([industry] * 4)
    
    # High Priority: Sub Company Vertical (weight: 3x)
    sub_vertical = str(row.get('Sub Company Vertical', '')).strip()
    if sub_vertical:
        content_parts.extend([sub_vertical] * 3)
    
    # Medium Priority: Company Business Model (weight: 2x)
    business_model = str(row.get('Company Business Model', '')).strip()
    if business_model:
        content_parts.extend([business_model] * 2)
    
    # Base Priority: Description (weight: 1x)
    description = str(row.get('Description', '')).strip()
    if description:
        content_parts.append(description)
    
    return ' '.join(content_parts)

def extract_enhanced_content_from_url(url):
    """
    Enhanced content extraction from URL focusing on actual page content:
    1. Brand/company name from meta tags and structured data
    2. Meta tags (title, description, keywords)
    3. Structured data (JSON-LD, microdata)
    4. Main content headings and text
    5. Category/navigation keywords
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_parts = []
        
        # 1. Extract brand/company name from various sources (highest weight: 5x)
        brand_selectors = [
            soup.find('meta', property='og:site_name'),
            soup.find('meta', {'name': 'application-name'}),
            soup.find('meta', {'name': 'apple-mobile-web-app-title'}),
            soup.find('span', {'itemprop': 'name'}),
            soup.find('h1', class_=re.compile(r'brand|logo|site-title', re.I))
        ]
        
        for selector in brand_selectors:
            if selector:
                brand = selector.get('content', '') if selector.get('content') else selector.get_text()
                if brand.strip():
                    content_parts.extend([brand.strip()] * 5)
                    print(f"🏢 Brand found: {brand.strip()}")
                    break
        
        # 2. Structured data (JSON-LD) - Very important for accurate brand/category info (weight: 4x)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts[:3]:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Extract organization name and description
                    org_name = data.get('name', '') or data.get('organizationName', '')
                    org_desc = data.get('description', '')
                    org_type = data.get('@type', '')
                    
                    if org_name:
                        content_parts.extend([org_name] * 4)
                        print(f"📋 Schema.org name: {org_name}")
                    if org_desc:
                        content_parts.extend([org_desc] * 3)
                    if org_type:
                        content_parts.extend([org_type] * 2)
            except:
                pass
        
        # 3. Meta tags (weight: 4x for title/description, 3x for keywords)
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            content_parts.extend([title_tag.get_text().strip()] * 4)
            print(f"📄 Title: {title_tag.get_text().strip()[:100]}")
        
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content', '').strip():
            content_parts.extend([og_title.get('content').strip()] * 4)
        
        description = soup.find('meta', {'name': 'description'})
        if description and description.get('content', '').strip():
            content_parts.extend([description.get('content').strip()] * 4)
            print(f"📝 Description: {description.get('content').strip()[:100]}")
        
        og_description = soup.find('meta', property='og:description')
        if og_description and og_description.get('content', '').strip():
            content_parts.extend([og_description.get('content').strip()] * 3)
        
        keywords = soup.find('meta', {'name': 'keywords'})
        if keywords and keywords.get('content', '').strip():
            content_parts.extend([keywords.get('content').strip()] * 3)
        
        # 4. Main content headings (weight: 3x for H1, 2x for H2/H3)
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags[:3]:  # Limit to first 3 H1s
            heading_text = h1.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 3)
        
        h2_h3_tags = soup.find_all(['h2', 'h3'])
        for heading in h2_h3_tags[:10]:  # Limit to first 10 H2/H3s
            heading_text = heading.get_text().strip()
            if heading_text and len(heading_text) > 3:
                content_parts.extend([heading_text] * 2)
        
        # 5. Category and navigation keywords (weight: 2x)
        nav_elements = soup.find_all(['nav', 'header'])
        for nav in nav_elements[:3]:
            nav_text = ' '.join([a.get_text().strip() for a in nav.find_all('a') if a.get_text().strip()])
            if nav_text:
                content_parts.extend([nav_text] * 2)
        
        # 6. Main content text (weight: 1x)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'role': 'main'})
        if main_content:
            paragraphs = main_content.find_all('p')
            for p in paragraphs[:15]:  # Limit to first 15 paragraphs
                text = p.get_text().strip()
                if len(text) > 50:  # Only meaningful paragraphs
                    content_parts.append(text)
        
        final_content = ' '.join(content_parts)
        print(f"📄 Extracted {len(final_content)} characters from URL")
        return final_content
        
    except Exception as e:
        print(f"❌ Error extracting content from URL: {e}")
        return ""

def calculate_combined_score(similarity_scores, trust_scores, similarity_weight=0.6, trust_weight=0.4):
    """
    Combine similarity scores and trust scores with configurable weights
    
    Args:
        similarity_scores: Array of TF-IDF similarity scores (0-1)
        trust_scores: Array of Trust Score ratings
        similarity_weight: Weight for similarity (default 0.6 = 60%)
        trust_weight: Weight for trust score (default 0.4 = 40%)
    
    Returns:
        Combined normalized scores
    """
    # Normalize trust scores to 0-1 range (assuming max trust score is 100)
    max_trust_score = 100
    normalized_trust = np.array(trust_scores) / max_trust_score
    
    # Normalize similarity scores (already in 0-1 range from cosine similarity)
    normalized_similarity = np.array(similarity_scores)
    
    # Calculate weighted combination
    combined_scores = (similarity_weight * normalized_similarity) + (trust_weight * normalized_trust)
    
    return combined_scores

def deduplicate_publishers(publishers):
    """
    Deduplicate publishers based on identical data across key fields.
    Keep the first occurrence (usually the longer/more complete name).
    """
    if not publishers:
        return publishers
    
    compare_fields = [
        'TRUST SCORE RATING',
        'Estimated Avg. Ahrefs DR',
        'Estimated Avg. Moz DA',
        'Estimated Avg. Semrush AS',
        'Website',
        'Associated Contact',
        'Description',
        'Industry Vertical',
        'Company Business Model',
        'Sub Company Vertical',
        'RBL Brand'
    ]
    
    seen_signatures = {}
    deduplicated = []
    removed_duplicates = []
    
    for pub in publishers:
        signature_parts = []
        for field in compare_fields:
            value = str(pub.get(field, '')).strip().lower()
            
            # Normalize URLs by removing www. and trailing slashes
            if field == 'Website':
                value = value.replace('www.', '').rstrip('/')
            
            signature_parts.append(value)
        
        signature = tuple(signature_parts)
        
        if signature not in seen_signatures:
            seen_signatures[signature] = pub
            deduplicated.append(pub)
        else:
            existing_pub = seen_signatures[signature]
            existing_name = existing_pub.get('Clean Up Name', '')
            current_name = pub.get('Clean Up Name', '')
            
            if len(current_name) > len(existing_name):
                idx = deduplicated.index(existing_pub)
                deduplicated[idx] = pub
                seen_signatures[signature] = pub
                removed_duplicates.append(f"Replaced '{existing_name}' with '{current_name}'")
            else:
                removed_duplicates.append(f"Removed duplicate '{current_name}' (kept '{existing_name}')")
    
    if removed_duplicates:
        print(f"🔄 Deduplication removed {len(removed_duplicates)} duplicates:")
        for msg in removed_duplicates[:5]:  # Show first 5
            print(f"  - {msg}")
    
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
    """Autocomplete endpoint for partner names with caching and debouncing"""
    try:
        query = request.args.get('query', '').strip().lower()
        
        # Require at least 2 characters
        if len(query) < 2:
            return jsonify([])
        
        # Use cached partner names if available
        if cache_data['partner_names'] is not None:
            partner_names = cache_data['partner_names']
        else:
            df = get_sheet_data()
            if df.empty:
                return jsonify([])
            partner_names = df['Clean Up Name'].dropna().astype(str).unique().tolist()
        
        # Filter suggestions
        suggestions = [name for name in partner_names if query in name.lower()]
        
        # Limit to top 20 for performance
        return jsonify(sorted(suggestions[:20]))
    
    except Exception as e:
        print(f"❌ Error in autocomplete: {e}")
        return jsonify([])

@app.route('/result', methods=['POST'])
def result():
    """Trust Score Lookup Result Page"""
    try:
        partner_input = request.form.get('partner_name', '').strip()
        print(f"🔍 User searched for: {partner_input}")
        
        df = get_sheet_data()
        if df.empty:
            return render_template('result_trust_score.html', error="Unable to load data from Google Sheets.")
        
        # Normalize for comparison
        df['partner_normalized'] = df['Clean Up Name'].str.strip().str.lower()
        partner_normalized = partner_input.lower()
        
        # Check if partner exists
        if partner_normalized not in df['partner_normalized'].values:
            return render_template('result_trust_score.html', error=f"Partner '{partner_input}' not found.")
        
        # Filter data for selected partner
        selected = df[df['partner_normalized'] == partner_normalized]
        if selected.empty:
            return render_template('result_trust_score.html', error="No data found for the selected partner.")
        
        # Calculate Trust Score (average if multiple rows)
        trust_score = selected['TRUST SCORE RATING'].mean()
        partner_display_name = selected['Clean Up Name'].iloc[0]
        
        # Aggregate Affiliate Brands data
        affiliate_brands = (
            selected.groupby('Affiliate_Brand', as_index=False)
            .agg({
                'total_revenue': 'sum',
                'total_order_action': 'sum',
                'total_clicks': 'sum',
                'CR': 'mean'
            })
        ).to_dict('records')
        
        # Find Similar Partners using weighted TF-IDF
        df['weighted_content'] = df.apply(create_weighted_content, axis=1)
        selected_content = create_weighted_content(selected.iloc[0])
        
        if selected_content.strip():
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                all_content = df['weighted_content'].fillna('').tolist() + [selected_content]
                tfidf_matrix = vectorizer.fit_transform(all_content)
                similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                
                df_copy = df.copy()
                df_copy['similarity'] = similarity_scores
                df_copy = df_copy[df_copy['partner_normalized'] != partner_normalized]
                
                # Calculate combined score with trust score consideration
                combined_scores = calculate_combined_score(
                    df_copy['similarity'].values,
                    df_copy['TRUST SCORE RATING'].values,
                    similarity_weight=0.7,  # 70% weight on similarity
                    trust_weight=0.3  # 30% weight on trust score
                )
                df_copy['combined_score'] = combined_scores
                
                # Sort by combined score
                similar_df = df_copy.sort_values(
                    by='combined_score',
                    ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(50)
                
                print(f"🎯 Top 5 combined scores: {similar_df['combined_score'].head().tolist()}")
            
            except Exception as e:
                print(f"⚠️ TF-IDF failed, using trust score fallback: {e}")
                similar_df = df[df['partner_normalized'] != partner_normalized].sort_values(
                    by='TRUST SCORE RATING', ascending=False
                ).drop_duplicates(subset=['Clean Up Name']).head(50)
        else:
            similar_df = df[df['partner_normalized'] != partner_normalized].sort_values(
                by='TRUST SCORE RATING', ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(50)
        
        # Prepare similar partners data
        similar_partners = similar_df[[
            'Clean Up Name', 'TRUST SCORE RATING', 'Estimated Avg. Ahrefs DR',
            'Estimated Avg. Moz DA', 'Estimated Avg. Semrush AS', 'Website',
            'Associated Contact', 'Description', 'Industry Vertical',
            'Company Business Model', 'Sub Company Vertical', 'RBL Brand'
        ]].to_dict('records')
        
        # Apply deduplication
        similar_partners = deduplicate_publishers(similar_partners)
        
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
    """Brand URL Lookup Result Page with Enhanced Matching and Trust Score Consideration"""
    try:
        brand_url = request.form.get('brand_url', '').strip()
        print(f"🔍 Brand URL lookup: {brand_url}")
        
        # Extract enhanced content from URL
        url_content = extract_enhanced_content_from_url(brand_url)
        if not url_content:
            return render_template('result_brand_lookup.html', error="Unable to extract content from the URL. Please check the URL and try again.")
        
        df = get_sheet_data()
        if df.empty:
            return render_template('result_brand_lookup.html', error="Unable to load data from Google Sheets.")
        
        # Create weighted content for matching
        df['weighted_content'] = df.apply(create_weighted_content, axis=1)
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1500,
                ngram_range=(1, 3),  # Include bigrams and trigrams for better matching
                min_df=1,
                max_df=0.95
            )
            all_content = df['weighted_content'].fillna('').tolist() + [url_content]
            tfidf_matrix = vectorizer.fit_transform(all_content)
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            df_copy = df.copy()
            df_copy['similarity'] = similarity_scores
            
            # Calculate combined score with trust score consideration
            combined_scores = calculate_combined_score(
                df_copy['similarity'].values,
                df_copy['TRUST SCORE RATING'].values,
                similarity_weight=0.65,  # 65% weight on content similarity
                trust_weight=0.35  # 35% weight on trust score
            )
            df_copy['combined_score'] = combined_scores
            
            # Sort by combined score
            publishers_df = df_copy.sort_values(
                by='combined_score',
                ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(50)
            
            print(f"🎯 Top 5 similarity scores: {publishers_df['similarity'].head().tolist()}")
            print(f"🏆 Top 5 trust scores: {publishers_df['TRUST SCORE RATING'].head().tolist()}")
            print(f"⭐ Top 5 combined scores: {publishers_df['combined_score'].head().tolist()}")
        
        except Exception as e:
            print(f"⚠️ TF-IDF failed in brand lookup, using trust score: {e}")
            publishers_df = df.sort_values(
                by='TRUST SCORE RATING', ascending=False
            ).drop_duplicates(subset=['Clean Up Name']).head(50)
        
        # Prepare publishers data
        publishers = publishers_df[[
            'Clean Up Name', 'TRUST SCORE RATING', 'Estimated Avg. Ahrefs DR',
            'Estimated Avg. Moz DA', 'Estimated Avg. Semrush AS', 'Website',
            'Associated Contact', 'Description', 'Industry Vertical',
            'Company Business Model', 'Sub Company Vertical', 'RBL Brand'
        ]].to_dict('records')
        
        # Apply deduplication
        publishers = deduplicate_publishers(publishers)
        
        return render_template('result_brand_lookup.html', publishers=publishers)
    
    except Exception as e:
        print(f"❌ Error in /publisher_lookup: {e}")
        return render_template('result_brand_lookup.html', error="An unexpected error occurred.")

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download partners data as CSV with user-friendly column names"""
    try:
        data = request.json.get('publishers', [])
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Create DataFrame from the data
        df_export = pd.DataFrame(data)
        
        # Rename columns to match the display names in the table
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
        
        # Rename columns that exist in the dataframe
        df_export = df_export.rename(columns=column_mapping)
        
        # Reorder columns to match table display order
        desired_order = [
            'Partner Name', 'Trust Partner Score', 'Avg. Ahrefs', 'Avg. Moz', 
            'Avg. Semrush', 'URL', 'Contact', 'Description', 
            'Industry Vertical', 'Business Model', 'Company Vertical', 'RBL Brand'
        ]
        
        # Only include columns that exist
        existing_cols = [col for col in desired_order if col in df_export.columns]
        df_export = df_export[existing_cols]
        
        # Create CSV
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