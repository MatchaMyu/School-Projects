# Notes:
# This script manages a database for customer transactions and product UPCs, integrating with the UPCItemDB API to fetch product provider information.
#
# - **Configuration Note**: Users should update the `path` and `MAX_UPC_CNT` variables to match their local file system and API plan.
#
# - **Data Import**: Imports customer and transaction data from Excel and CSV files, ensuring all columns are stored as strings to avoid 
#                    decimal/scientific notation issues.
#                  
# - **Database Setup**: Creates SQLite tables (Customers, Transactions, ProductProviders, UPC_Offers) to store customer data, transaction details, 
#                       provider counts, and offer details.
#                   
# - **UPC Validation**: Validates UPC codes to ensure they are 12, 13, or 14-digit numbers, padding 10 or 11-digit UPCs with leading zeros as needed.
#
# - **API Integration**: Queries the UPCItemDB API in batches to retrieve provider and offer information for valid UPCs, respecting rate limits 
#                        (15 calls per 30 seconds) and a maximum UPC count of 10 per batch. Script auto cancels upon reaching maximum request limit.
#                     
# - **Error Handling**: Implements retry logic with exponential backoff for API failures (e.g., HTTP 429 errors) and handles invalid UPCs or API errors.
#
# - **Data Storage**: Saves API responses, provider counts, and offer details to the database, calculating the difference between API-provided and 
#                     customer-provided providers.
#
# - **Main Workflow**: Processes unique UPCs from the Transactions table, skipping those already in ProductProviders, and updates the database with 
#                      provider and offer data.

import pandas as pd
import sqlite3
import urllib.request
import json
import time
import gzip
import os
import uuid
from urllib.error import HTTPError
from datetime import datetime

path = r"C:\Users\A81383\OneDrive - Andersen Corporation\Documents\Documents\_School\Fall 2025\DATA 499\Capstone_2025-main"
MAX_UPC_CNT = 10000 # MAX_UPC_CNT = 10,000 | 10,000 UPCs | 1,000 API calls | 2.2 seconds per call equals approximate 36 minute runtime

API_BASE_URL = 'https://api.upcitemdb.com/prod/v1/lookup' #'https://api.upcitemdb.com/prod/trail/lookup'
USER_KEY = '5334e17227c6d5bed019a62976325d9b'

RATE_LIMIT_SLEEP = 2.2 # Max 15 calls per 30 seconds. Added extra .2
MAX_RETRIES = 3
RETRY_WAIT = 5
EXPONENTIAL_BACKOFF_FACTOR = 2
BATCH_SIZE = 10 # 10 for v1 endpoint, 2 for trial endpoint
UPC_CNT = 0
db_path = os.path.join(path, 'SPS_DB.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS Customers (
    "ORG ID" TEXT,
    "First Subscription Date" TEXT,
    "Doc Flow Total Monthly Average" TEXT,
    "Industry SPS" TEXT,
    "AnnualRevenue" TEXT,
    "Revenue Range" TEXT,
    "Industry Segment" TEXT,
    "Industry Vertical" TEXT,
    "Total Retailer Connections" TEXT,
    "years_in_business" TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS Transactions (
    "FirstParcelDate" TEXT,
    "ConsumerPackageCode" TEXT,
    "ORG_ID" TEXT,
    "GTIN" TEXT,
    "UPCCaseCode" TEXT,
    FOREIGN KEY (ORG_ID) REFERENCES Customers(ORG_ID)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS ProductProviders(
    ProductProviders_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ConsumerPackageCode TEXT UNIQUE NOT NULL,
    Count_Ubcitemdb_Providers INT,
    Count_Customers_Providers INT,
    Providers_Difference INT,
    ApiResponse TEXT,
    Request_URL TEXT,
    Raw_URL_Response TEXT,
    FOREIGN KEY (ConsumerPackageCode) REFERENCES Transactions(ConsumerPackageCode)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS UPC_Offers (
    Offer_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ConsumerPackageCode TEXT NOT NULL,
    Store TEXT,
    Product_Info TEXT,
    Price TEXT,
    Last_Updated TEXT,
    FOREIGN KEY (ConsumerPackageCode) REFERENCES ProductProviders(ConsumerPackageCode),
    UNIQUE (ConsumerPackageCode, Store, Product_Info, Price, Last_Updated)
)
''')

def AddProductProvidersUpcOffersData():
    try:
        product_file = os.path.join(path, "ProductProviders.xlsx")
        offers_file = os.path.join(path, "UPC_Offers.xlsx")

        if not os.path.exists(product_file):
            print(f"Error: ProductProviders file not found at {product_file}")
            return
        if not os.path.exists(offers_file):
            print(f"Error: UPC_Offers file not found at {offers_file}")
            return

        cursor.execute("SELECT ConsumerPackageCode FROM Transactions WHERE ConsumerPackageCode IS NOT NULL")
        valid_transaction_upcs = {row[0] for row in cursor.fetchall()}
        if not valid_transaction_upcs:
            print("Error: No valid ConsumerPackageCode values found in Transactions table")
            return

        product_df = pd.read_excel(product_file, dtype=str)

        if 'ProductProviders_ID' in product_df.columns:
            product_df = product_df.drop(columns=['ProductProviders_ID'])

        numeric_columns = ['Count_Ubcitemdb_Providers', 'Count_Customers_Providers', 'Providers_Difference']
        for col in numeric_columns:
            if col in product_df.columns:
                product_df[col] = pd.to_numeric(product_df[col], errors='coerce').fillna(0).astype(int)

        invalid_product_upcs = set(product_df['ConsumerPackageCode']) - valid_transaction_upcs
        if invalid_product_upcs:
            print(f"Warning: {len(invalid_product_upcs)} ConsumerPackageCode values in ProductProviders.xlsx not in Transactions: {list(invalid_product_upcs)[:10]}")
        product_df = product_df[product_df['ConsumerPackageCode'].isin(valid_transaction_upcs)]

        duplicates = product_df.duplicated(subset=['ConsumerPackageCode'], keep=False)
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate ConsumerPackageCode values in ProductProviders.xlsx:")
            print(product_df[duplicates][['ConsumerPackageCode']].head())
            product_df = product_df.drop_duplicates(subset=['ConsumerPackageCode'])

        if not product_df.empty:
            product_df.to_sql('ProductProviders', conn, if_exists='replace', index=False)
            print(f"Loaded ProductProviders from {product_file} ({len(product_df)} records)")
        else:
            print("Warning: No valid ProductProviders records to import after filtering")

        offers_df = pd.read_excel(offers_file, dtype=str)

        if 'Offer_ID' in offers_df.columns:
            offers_df = offers_df.drop(columns=['Offer_ID'])

        cursor.execute("SELECT ConsumerPackageCode FROM ProductProviders")
        valid_product_upcs = {row[0] for row in cursor.fetchall()}
        if not valid_product_upcs:
            print("Error: No valid ConsumerPackageCode values found in ProductProviders table")
            return

        invalid_offer_upcs = set(offers_df['ConsumerPackageCode']) - valid_product_upcs
        if invalid_offer_upcs:
            print(f"Warning: {len(invalid_offer_upcs)} ConsumerPackageCode values in UPC_Offers.xlsx not in ProductProviders: {list(invalid_offer_upcs)[:10]}")
        offers_df = offers_df[offers_df['ConsumerPackageCode'].isin(valid_product_upcs)]

        unique_cols = ['ConsumerPackageCode', 'Store', 'Product_Info', 'Price', 'Last_Updated']
        duplicates = offers_df.duplicated(subset=unique_cols, keep=False)
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate rows in UPC_Offers.xlsx for UNIQUE constraint:")
            print(offers_df[duplicates][unique_cols].head())
            offers_df = offers_df.drop_duplicates(subset=unique_cols)

        if not offers_df.empty:
            offers_df.to_sql('UPC_Offers', conn, if_exists='replace', index=False)
            print(f"Loaded UPC_Offers from {offers_file} ({len(offers_df)} records)")
        else:
            print("Warning: No valid UPC_Offers records to import after filtering")

        conn.commit()
        print("Successfully imported ProductProviders and UPC_Offers tables")

    except Exception as e:
        print(f"Error loading Excel tables: {type(e).__name__}: {str(e)}")
        raise
        
def AddCustomerTransactionData():
    customers_df = pd.read_excel(os.path.join(path, 'Customers.xlsx'))
    transactions_df = pd.read_csv(os.path.join(path, 'Transactions.csv'))

    for col in transactions_df.columns:
        transactions_df[col] = transactions_df[col].astype(str).str.replace(r'\.0+$', '', regex=True).str.strip()

    customers_df = customers_df.astype(str)
    
    customers_df.to_sql('Customers', conn, if_exists='replace', index=False)
    transactions_df.to_sql('Transactions', conn, if_exists='replace', index=False)

    # transactions_check = pd.read_sql_query("SELECT * FROM Transactions LIMIT 5", conn)

    # for col in transactions_check.columns:
    #     has_decimals = transactions_check[col].str.contains(r'\.\d', regex=True, na=False).any()
    #     print(f"Column '{col}' contains decimals: {has_decimals}")

    conn.commit()
    conn.close()

    print("Tables created and data from Customers.xlsx and Transactions.csv has been successfully saved to SPS_DB.db with all columns as strings, avoiding decimals and scientific notation.")
    
def export_to_excel():
    try:
        conn = sqlite3.connect(db_path)

        product_providers_df = pd.read_sql_query("SELECT * FROM ProductProviders", conn)
        upc_offers_df = pd.read_sql_query("SELECT * FROM UPC_Offers", conn)

        conn.close()
 
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        product_providers_file = os.path.join(path, f'ProductProviders_{timestamp}.xlsx')
        upc_offers_file = os.path.join(path, f'UPC_Offers_{timestamp}.xlsx')

        product_providers_df.to_excel(product_providers_file, index=False)
        print(f"Exported ProductProviders to {product_providers_file}")
        
        upc_offers_df.to_excel(upc_offers_file, index=False)
        print(f"Exported UPC_Offers to {upc_offers_file}")
    except Exception as e:
        print(f"Error exporting tables to Excel: {e}")
        
def validate_upc(upc):
    if not upc or upc == '0' or not upc.isdigit():
        #print(f"UPC {upc} - Invalid: Empty, zero, or non-numeric")
        return False
    if upc and len(upc) == 10:
        upc = '00' + upc
    if upc and len(upc) == 11:
        upc = '0' + upc
    if not upc or not upc.isdigit() or len(upc) not in (12, 13, 14):
        #print(f"UPC {upc} - Invalid: Not a 12, 13, or 14-digit number")
        return False
    return True

def lookup_upc_batch(upcs):
    global UPC_CNT
    valid_upcs = [upc for upc in upcs if validate_upc(upc)]
    invalid_upcs = [upc for upc in upcs if upc not in valid_upcs]
    result = {upc: {"upc": upc, "status": "invalid_format"} for upc in invalid_upcs}
    if not valid_upcs:
        print(f"Batch UPCs {upcs} - All UPCs invalid, skipping")
        return result, None, None
    if UPC_CNT + len(valid_upcs) > MAX_UPC_CNT:
        print(f"API call limit of {MAX_UPC_CNT} would be exceeded with {len(valid_upcs)} UPCs")
        return {**result, **{upc: {"upc": upc, "status": "api_limit_reached"} for upc in valid_upcs}}, None, None
    upc_string = ','.join(valid_upcs)
    params = f'?upc={upc_string}'
    url = API_BASE_URL + params
    headers = {
        'accept': 'application/json',
        'user_key': USER_KEY,
        'key_type': '3scale',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Encoding': 'gzip, deflate'
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req) as response:
                content_encoding = response.headers.get('Content-Encoding', '')
                raw_data = response.read()
                if 'gzip' in content_encoding.lower():
                    raw_data = gzip.decompress(raw_data)
                decoded_response = raw_data.decode('utf-8')
                json_data = json.loads(decoded_response)
                remaining = int(response.headers.get('X-RateLimit-Remaining', -1))
                UPC_CNT += len(valid_upcs)
                print(f"Batch UPCs {upc_string} - Success, Remaining requests: {remaining}")
                if remaining == 0:
                    print(f"Rate limit exhausted (0 requests remaining), stopping further API calls")
                    UPC_CNT = MAX_UPC_CNT
                response_file = os.path.join(path, f'batch_response_{uuid.uuid4()}.txt')
                return {**result, **json_data}, url, decoded_response
        except HTTPError as e:
            UPC_CNT += len(valid_upcs)
            error_response = None
            try:
                error_response = e.read()
                if 'gzip' in e.headers.get('Content-Encoding', '').lower():
                    error_response = gzip.decompress(error_response)
                error_response = error_response.decode('utf-8')
            except:
                pass
            if e.code == 429:
                wait_time = RETRY_WAIT * (EXPONENTIAL_BACKOFF_FACTOR ** attempt)
                print(f"Batch UPCs {upc_string} - Rate limit exceeded, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            elif e.code in (403, 401, 404):
                print(f"Batch UPCs {upc_string} - HTTP Error {e.code}: {e.reason}")
                return {**result, **{upc: {"upc": upc, "status": f"http_error_{e.code}", "error": error_response} for upc in valid_upcs}}, url, error_response
            else:
                print(f"Batch UPCs {upc_string} - HTTP Error {e.code}: {e.reason}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_WAIT)
                continue
        except Exception as e:
            UPC_CNT += len(valid_upcs)
            print(f"Batch UPCs {upc_string} - Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
                continue
            return {**result, **{upc: {"upc": upc, "status": "error", "message": str(e)} for upc in valid_upcs}}, url, None
    print(f"Batch UPCs {upc_string} - Failed after {MAX_RETRIES} attempts")
    return {**result, **{upc: {"upc": upc, "status": "failed"} for upc in valid_upcs}}, url, None

def test_single_upc(upc):
    global UPC_CNT
    if not validate_upc(upc):
        print(f"UPC {upc} - Invalid UPC, skipping")
        return

    if UPC_CNT >= MAX_UPC_CNT:
        print(f"UPC {upc} - API call limit reached")
        cursor.execute('''
            INSERT OR IGNORE INTO ProductProviders
            (ConsumerPackageCode, Count_Ubcitemdb_Providers, Count_Customers_Providers, Providers_Difference, ApiResponse, Request_URL, Raw_URL_Response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (upc, 0, 0, 0, json.dumps({"upc": upc, "status": "api_limit_reached"}), None, None))
        conn.commit()
        return

    result, request_url, raw_response = lookup_upc_batch([upc])
    customer_query = '''
    SELECT COUNT(DISTINCT ORG_ID)
    FROM Transactions
    WHERE ConsumerPackageCode = ?
    '''
    count_customers = cursor.execute(customer_query, (upc,)).fetchone()[0]
    count_ubc = 0
    api_response = json.dumps({"upc": upc, "status": "not_processed"})

    if isinstance(result, dict) and upc in result and result[upc].get("status") == "invalid_format":
        print(f"UPC {upc} - Invalid format, no providers found")
        api_response = json.dumps(result[upc])
    elif result and result.get('code') == 'OK':
        items = result.get('items', [])
        for item in items:
            item_upc = str(item.get('upc', '')) if item.get('upc') else ''
            item_ean = str(item.get('ean', '')) if item.get('ean') else ''
            item_asin = str(item.get('asin', '')) if item.get('asin') else ''
            if str(upc).lstrip('0') in (item_upc.lstrip('0'), item_ean.lstrip('0'), item_asin.lstrip('0')):
                offers = item.get('offers', [])
                unique_merchants = {offer.get('merchant') for offer in offers if offer.get('merchant')}
                count_ubc = len(unique_merchants)
                api_response = json.dumps(item)
                if count_ubc > 0:
                    print(f"UPC {upc} - Found {count_ubc} providers: {unique_merchants}")
                    for offer in offers:
                        store = offer.get('merchant')
                        product_info = offer.get('title')
                        price = offer.get('price')
                        last_updated = offer.get('updated_t')
                        if last_updated is None:
                            last_updated = datetime.now().isoformat()
                        if store and product_info and price and last_updated:
                            cursor.execute('''
                                INSERT OR IGNORE INTO UPC_Offers
                                (ConsumerPackageCode, Store, Product_Info, Price, Last_Updated)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (upc, store, product_info, str(price), last_updated))
                # else:
                #     print(f"UPC {upc} - No providers found")
                break
        # else:
        #     print(f"UPC {upc} - No matching item in API response")
            api_response = json.dumps({"upc": upc, "status": "not_found_in_response"})
    else:
        print(f"UPC {upc} - API error: {result.get('status', 'unknown')}")
        api_response = json.dumps({"upc": upc, "status": result.get('status', 'error')})

    difference = max(0, count_ubc - count_customers)
    cursor.execute('''
        INSERT OR IGNORE INTO ProductProviders
        (ConsumerPackageCode, Count_Ubcitemdb_Providers, Count_Customers_Providers, Providers_Difference, ApiResponse, Request_URL, Raw_URL_Response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (upc, count_ubc, count_customers, difference, api_response, request_url, raw_response))
    conn.commit()
    print(f"UPC {upc} - UBC Providers: {count_ubc}, Customers Providers: {count_customers}")

def main():
    global UPC_CNT
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ProductProviders'")
    table_exists = cursor.fetchone() is not None
    if table_exists:
        query = '''
        SELECT DISTINCT ConsumerPackageCode
        FROM Transactions
        WHERE ConsumerPackageCode IS NOT NULL
        AND ConsumerPackageCode NOT IN (
            SELECT ConsumerPackageCode
            FROM ProductProviders
        )'''
    else:
        query = '''
        SELECT DISTINCT ConsumerPackageCode
        FROM Transactions
        WHERE ConsumerPackageCode IS NOT NULL
        '''
    
    unique_upcs = [row[0] for row in cursor.execute(query).fetchall() if validate_upc(row[0])]
    print(f"Found {len(unique_upcs)} unique UPCs: {unique_upcs[:10]}...")
    for i in range(0, len(unique_upcs), BATCH_SIZE):
        if UPC_CNT >= MAX_UPC_CNT:
            print(f"API call limit of {MAX_UPC_CNT} reached")
            break
        batch_upcs = unique_upcs[i:i + BATCH_SIZE]
        #print(f"Processing batch {i//BATCH_SIZE + 1}/{len(unique_upcs)//BATCH_SIZE + 1}: {batch_upcs}")
        result, request_url, raw_response = lookup_upc_batch(batch_upcs)
        if all(upc in result and result[upc].get("status") == "api_limit_reached" for upc in batch_upcs):
            #print(f"Batch {i//BATCH_SIZE + 1} skipped due to API limit")
            for upc in batch_upcs:
                cursor.execute('''
                    INSERT OR IGNORE INTO ProductProviders
                    (ConsumerPackageCode, Count_Ubcitemdb_Providers, Count_Customers_Providers, Providers_Difference, ApiResponse, Request_URL, Raw_URL_Response)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (upc, 0, 0, 0, json.dumps({"upc": upc, "status": "api_limit_reached"}), None, None))
            conn.commit()
            break
        upc_to_item = {}
        unmatched_upcs = set(batch_upcs)
       # print(f"Unmatched UPCs before processing: {unmatched_upcs}")
        if result and result.get('code') == 'OK':
            items = result.get('items', [])
            for item in items:
                item_upc = str(item.get('upc', '')) if item.get('upc') else ''
                item_ean = str(item.get('ean', '')) if item.get('ean') else ''
                item_asin = str(item.get('asin', '')) if item.get('asin') else ''
                for upc in batch_upcs:
                    if str(upc).lstrip('0') in (item_upc.lstrip('0'), item_ean.lstrip('0'), item_asin.lstrip('0')):
                        if upc in unmatched_upcs:  # Prevent KeyError
                            upc_to_item[upc] = item
                            unmatched_upcs.remove(upc)
                            #print(f"Matched UPC {upc}, remaining unmatched: {unmatched_upcs}")
                            break
        for upc in batch_upcs:
            if not validate_upc(upc):
                print(f"UPC {upc} - Skipping invalid UPC in processing loop")
                continue
            customer_query = '''
            SELECT COUNT(DISTINCT ORG_ID)
            FROM Transactions
            WHERE ConsumerPackageCode = ?
            '''
            count_customers = cursor.execute(customer_query, (upc,)).fetchone()[0]
            count_ubc = 0
            api_response = json.dumps({"upc": upc, "status": "not_processed"})
            if isinstance(result, dict) and upc in result and result[upc].get("status") == "invalid_format":
                print(f"UPC {upc} - Invalid format, no providers found")
                api_response = json.dumps(result[upc])
            elif upc in upc_to_item:
                item = upc_to_item[upc]
                offers = item.get('offers', [])
                unique_merchants = {offer.get('merchant') for offer in offers if offer.get('merchant')}
                count_ubc = len(unique_merchants)
                api_response = json.dumps(item)
                if count_ubc > 0:
                    #print(f"UPC {upc} - Found {count_ubc} providers: {unique_merchants}")
                    for offer in offers:
                        store = offer.get('merchant')
                        product_info = offer.get('title')
                        price = offer.get('price')
                        last_updated = offer.get('updated_t')
                        if last_updated is None:
                            last_updated = datetime.now().isoformat()
                        if store and product_info and price and last_updated:
                            cursor.execute('''
                                INSERT OR IGNORE INTO UPC_Offers
                                (ConsumerPackageCode, Store, Product_Info, Price, Last_Updated)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (upc, store, product_info, str(price), last_updated))
            else:
                # print(f"UPC {upc} - No matching item in API response")
                api_response = json.dumps({"upc": upc, "status": "not_found_in_response"})
            difference = max(0, count_ubc - count_customers)
            cursor.execute('''
                INSERT OR IGNORE INTO ProductProviders
                (ConsumerPackageCode, Count_Ubcitemdb_Providers, Count_Customers_Providers, Providers_Difference, ApiResponse, Request_URL, Raw_URL_Response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (upc, count_ubc, count_customers, difference, api_response, request_url, raw_response))
        conn.commit()
        
        if i + BATCH_SIZE < len(unique_upcs):
            time.sleep(RATE_LIMIT_SLEEP)
    print(f"Processing complete. Total API calls: {UPC_CNT}/{MAX_UPC_CNT}")
    conn.commit()

if __name__ == "__main__":
    # AddCustomerTransactionData()
    
    # cursor.execute("DELETE FROM ProductProviders")
    # cursor.execute("DELETE FROM UPC_Offers")
    # conn.commit()
    
    # AddProductProvidersUpcOffersData()
    
    main()
    conn.close()
    
    # export_to_excel()
