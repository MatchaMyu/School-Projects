import requests
import json
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

# --- Config ---
API_KEY = "REMOVED" 
ORG_ID = "REMOVED" 
INPUT_FILE = f"cpcs_for_org_{ORG_ID}.csv"
OUTPUT_FILE = f"api_results_org_{ORG_ID}.csv"
OUTPUT_FILE_OFFERS = f"api_offers_org_{ORG_ID}.csv"

# --- Load CPCs from CSV ---
print(f"✓ Loading CPCs from {INPUT_FILE}...")
input_df = pd.read_csv(INPUT_FILE)
cpc_list = input_df['ConsumerPackageCode'].tolist()
print(f"  Found {len(cpc_list)} CPCs to look up\n")


# --- API Call Function ---
def lookup_cpc(cpc):
    """Look up a single CPC via the API"""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip,deflate',
        'user_key': API_KEY,
        'key_type': '3scale'
    }

    url = f'https://api.upcitemdb.com/prod/v1/lookup?upc={cpc}'

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return json.loads(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error for CPC {cpc}: {e}")
        return None


# --- Process All CPCs ---
item_results = []
offer_results = []

print(f"✓ Looking up {len(cpc_list)} CPCs via API for ORG ID {ORG_ID}...\n")

for i, cpc in enumerate(cpc_list, 1):
    print(f"[{i}/{len(cpc_list)}] Fetching CPC: {cpc}")

    data = lookup_cpc(cpc)

    if data and 'items' in data and len(data['items']) > 0:
        for item in data['items']:
            # Basic item info
            item_results.append({
                'ORG_ID': ORG_ID,
                'CPC': cpc,
                'EAN': item.get('ean', ''),
                'Title': item.get('title', ''),
                'Brand': item.get('brand', ''),
                'Category': item.get('category', ''),
                'Description': item.get('description', ''),
                'Model': item.get('model', ''),
                'Lowest_Price': item.get('lowest_recorded_price', ''),
                'Highest_Price': item.get('highest_recorded_price', ''),
                'Currency': item.get('currency', 'USD'),
                'Number_of_Offers': len(item.get('offers', [])),
            })

            # Detailed offer information
            offers = item.get('offers', [])
            if offers:
                print(f"  ✓ Found {len(offers)} offers")
                for offer in offers:
                    updated_timestamp = offer.get('updated_t', 0)
                    updated_date = datetime.fromtimestamp(updated_timestamp).strftime(
                        '%Y-%m-%d %H:%M:%S') if updated_timestamp else 'Unknown'

                    offer_results.append({
                        'ORG_ID': ORG_ID,
                        'CPC': cpc,
                        'EAN': item.get('ean', ''),
                        'Item_Title': item.get('title', ''),
                        'Brand': item.get('brand', ''),
                        'Merchant': offer.get('merchant', ''),
                        'Domain': offer.get('domain', ''),
                        'Offer_Title': offer.get('title', ''),
                        'Currency': offer.get('currency', 'USD'),
                        'List_Price': offer.get('list_price', ''),
                        'Sale_Price': offer.get('price', ''),
                        'Shipping': offer.get('shipping', ''),
                        'Condition': offer.get('condition', ''),
                        'Availability': offer.get('availability', 'Available'),
                        'Link': offer.get('link', ''),
                        'Last_Updated': updated_date,
                        'Updated_Timestamp': updated_timestamp,
                    })
            else:
                print(f"  ✓ Found item but no current offers")
    else:
        item_results.append({
            'ORG_ID': ORG_ID,
            'CPC': cpc,
            'EAN': '',
            'Title': 'NOT FOUND',
            'Brand': '',
            'Category': '',
            'Description': '',
            'Model': '',
            'Lowest_Price': '',
            'Highest_Price': '',
            'Currency': '',
            'Number_of_Offers': 0,
        })
        print(f"  ⚠️  No data found")


    time.sleep(0.5)

# --- Save to CSV files ---
items_df = pd.DataFrame(item_results)
items_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {len(item_results)} item results to {OUTPUT_FILE}")

if offer_results:
    offers_df = pd.DataFrame(offer_results)
    offers_df.to_csv(OUTPUT_FILE_OFFERS, index=False)
    print(f"✓ Saved {len(offer_results)} offer details to {OUTPUT_FILE_OFFERS}")

# --- Summary ---
found = sum(1 for r in item_results if r['Title'] != 'NOT FOUND')
not_found = len(item_results) - found
total_offers = len(offer_results)

print(f"\n Summary for ORG ID {ORG_ID}:")
print(f"  CPCs queried: {len(cpc_list)}")
print(f"  Items found: {found}")
print(f"  Items not found: {not_found}")
print(f"  Total offers: {total_offers}")

if offer_results:
    merchant_counts = offers_df['Merchant'].value_counts()
    print(f"\n Top Merchants:")
    for merchant, count in merchant_counts.head(5).items():
        print(f"  {merchant}: {count} offers")