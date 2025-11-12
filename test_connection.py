import gspread
from oauth2client.service_account import ServiceAccountCredentials

def test_sheet_connection():
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds_path = "credentials.json"
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        
        # Open the sheet
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1b4_yuhEeLN-u21KHLEJOenDa1EdG6iQXpUi8ASXYZHk/edit?gid=1170846120"
        ).worksheet("Render Upload")
        
        data = sheet.get_all_records()
        print(f"✅ Successfully connected to Google Sheets!")
        print(f"📊 Found {len(data)} records")
        
        if data:
            print("\n📋 Column names found:")
            columns = list(data[0].keys())
            for i, col in enumerate(columns):
                print(f"   {i+1:2d}. '{col}'")
                
            print(f"\n📄 Sample data from first row:")
            first_row = data[0]
            for key, value in list(first_row.items())[:8]:  # Show first 8 fields
                print(f"   {key}: {str(value)[:50]}...")
                
        return True, columns if data else []
        
    except Exception as e:
        print(f"❌ Error connecting to Google Sheets: {e}")
        return False, []

if __name__ == "__main__":
    success, columns = test_sheet_connection()
    if success:
        print(f"\n🎉 Ready to update app.py with correct column names!")