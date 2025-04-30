import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import pandas as pd
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import os
from rapidfuzz import fuzz, process
import traceback

# --- Default Comparison Thresholds (Fixed as requested) ---
DEFAULT_ITEM_NAME_THRESHOLD = 90
DEFAULT_CATEGORY_THRESHOLD = 90
DEFAULT_VERY_HIGH_NAME_THRESHOLD = 95 # Threshold for considering names 'very similar'

# --- Selenium Setup (for dynamic content) ---
# Global variable for driver path (optional, if not in PATH)
# CHROME_DRIVER_PATH = "/path/to/chromedriver" # Uncomment and set if needed

@st.cache_resource
def get_chrome_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox') # Recommended for headless in some environments
    options.add_argument('--disable-dev-shm-usage') # Recommended for headless
    options.add_argument('--disable-gpu') # Might be necessary in some environments
    options.add_argument('--window-size=1920x1080') # Good standard size
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-extensions')
    # Add a user agent to avoid detection
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_argument('--log-level=3') # Suppress excessive logs
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Suppress Chrome log messages

    # Use pre-installed Chrome/driver path if available (e.g. on Streamlit Community Cloud)
    # Or ensure chromedriver is in PATH
    chrome_binary_location = os.environ.get("CHROME_BINARY_LOCATION")
    if chrome_binary_location:
         options.binary_location = chrome_binary_location

    try:
        # Use Service if specifying path, otherwise let it find chromedriver in PATH
        # service = Service(CHROME_DRIVER_PATH if 'CHROME_DRIVER_PATH' in globals() and CHROME_DRIVER_PATH else None)
        # Note: Let selenium-manager handle driver download/detection if not specified
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        st.error(f"Could not initialize Selenium WebDriver. Make sure ChromeDriver is installed and in your system's PATH. Error: {e}")
        st.info("See https://chromedriver.chromium.org/downloads for installation or check environment variables for deployment.")
        st.text(traceback.format_exc())
        return None

# Fetch JS-rendered content
def fetch_dynamic_content(url):
    st.info("Initializing Selenium WebDriver...")
    driver = get_chrome_driver()
    if driver is None:
        st.error("Selenium WebDriver failed to initialize.")
        return None

    try:
        st.info(f"Attempting to load dynamic content from {url}...")
        driver.get(url)
        # Give the page time to load JS content
        # Adjust sleep time based on typical page load complexity
        load_time = 7 # Increased wait time, adjust if needed
        st.info(f"Waiting {load_time} seconds for dynamic content to load...")
        time.sleep(load_time)

        page_source = driver.page_source
        st.info("Page source fetched.")
        return page_source
    except Exception as e:
        st.error(f"Error fetching dynamic content with Selenium: {e}")
        st.text(traceback.format_exc())
        return None
    finally:
        # Note: driver.quit() is commented out as get_chrome_driver is cached.
        # The driver instance persists across reruns unless the cache is cleared.
        # If you need to ensure the browser closes, manage the driver instance outside @st.cache_resource
        # or add explicit cleanup logic. For typical Streamlit Cloud usage, this might be okay.
        st.info("Selenium WebDriver operation finished.")


# Simple heuristic to guess if dynamic fetching might be needed
def should_attempt_dynamic_fetch(html_content):
    if not html_content:
        st.info("Static HTML fetch returned no content. Suggests dynamic content.")
        return True # If static fetch got nothing, definitely try dynamic

    soup = BeautifulSoup(html_content, "html.parser")

    # Check for common signs of JS rendering: absence of body content, script heavy
    text_length = len(soup.get_text(strip=True))
    body = soup.find('body')
    if body and not body.get_text(strip=True):
        st.info("Static HTML body is empty. Suggests dynamic content.")
        return True

    # A very basic check on the amount of text
    # This threshold might need tuning depending on typical website structures
    if text_length < 200:
        st.info(f"Static text content ({text_length} chars) seems sparse. Suggests dynamic content.")
        return True

    # Could add checks for lots of script tags, specific framework attributes etc.
    # For now, these basic checks are often sufficient.
    return False

# Pass the API key obtained from UI
def scrape_menu_with_gemini_from_url(url, api_key):
    st.info("Attempting to fetch content...")
    html = None
    try:
        # Initial static fetch
        st.info(f"Attempting static fetch from {url}...")
        response = requests.get(url, timeout=20) # Reduced timeout slightly
        response.raise_for_status() # Raise an exception for bad status codes
        html = response.text
        st.info("Static fetch successful.")

    except requests.exceptions.Timeout:
         st.warning(f"Static fetch timed out from {url}.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Error during static fetch from {url}: {e}")

    # Decide whether to attempt dynamic fetching based on initial content or error
    # Check if html is None or if the content seems dynamically loaded
    if html is None or should_attempt_dynamic_fetch(html):
        st.info("Attempting dynamic fetch...")
        dynamic_html = fetch_dynamic_content(url)
        if dynamic_html:
            html = dynamic_html
            st.info("Dynamic content fetched.")
        elif html is None:
             st.error("Both static and dynamic fetch failed to retrieve content.")
             return None
        else:
             # If dynamic fetch failed but static content exists, use static
             st.warning("Dynamic fetch failed or returned no content. Using static content.")


    if not html:
         st.error(f"Failed to retrieve any content from {url}")
         return None

    soup = BeautifulSoup(html, "html.parser")

    # Extract text, trying to preserve some structure with newlines
    raw_text = soup.get_text(separator='\n', strip=True)

    # Basic cleaning - remove excessive consecutive newlines and leading/trailing whitespace lines
    raw_text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())

    if not raw_text or len(raw_text.strip()) < 50:
        st.warning("Extracted text content is very short. Gemini may struggle to find a menu.")
        if len(raw_text.strip()) < 10: # Add a check for truly empty
             st.error("Less than 10 characters of text extracted. Cannot proceed.")
             return None


    # Limit raw_text length to manage token limits, especially for flash models
    max_chars = 25000 # Increased limit slightly, but still cautious
    if len(raw_text) > max_chars:
        st.warning(f"Raw text exceeds {max_chars} characters. Truncating for Gemini.")
        raw_text = raw_text[:max_chars] + "\n... [Content truncated]"


    prompt = f"""Extract food categories, items, prices, and any associated dietary information (like vegan, vegetarian, Jain) or allergy information (like nut-free, gluten-free) from the following text which is likely a restaurant menu or food-related page content.

    Pay close attention to the structure and layout to identify categories and their items. Associate prices correctly with items. Extract dietary/allergy info only if it appears directly next to or clearly associated with an item.

    For each item, if dietary or allergy information is present, include it as a list of strings for the keys "diet_types" and "allergy_info". If no information is found for an item, the lists should be empty (e.g., "diet_types": [], "allergy_info": []). Price should be extracted as a string including currency symbol if present. If a price is not explicitly listed but implied (e.g., "Ask"), capture that string. If no price is found or it's unclear, use "N/A".

    If you cannot confidently identify a menu structure or items, return an empty JSON object like {{"menu_data": []}}.

    Format the output strictly as a JSON object following this structure:
    {{
        "menu_data": [
            {{
                "category_name": "Category Name",
                "items": [
                    {{
                        "item_name": "Item Name",
                        "price": "Price String",
                        "diet_types": ["Diet Type 1", "Diet Type 2"],
                        "allergy_info": ["Allergy Info 1", "Allergy Info 2"]
                    }},
                    // ... other items in this category ...
                ]
            }},
            // ... other categories ...
        ]
    }}
    Only return the JSON object. Do not include any introductory or concluding text, explanations, or formatting like markdown code blocks (```json ```). Ensure the output is valid JSON parseable by json.loads().

    Menu Text:
    ---
    {raw_text}
    ---
    """
    st.info("Sending text to Gemini for extraction...")

    try:
        # Configure genai using the provided API key *before* making the call
        genai.configure(api_key=api_key) # Use the key from UI

        # Use a multimodal model for PDF/Image
        # MULTIMODAL_MODEL = "gemini-1.5-flash-latest" # Use latest version
        TEXT_MODEL = "gemini-1.5-flash-latest" # Use flash for speed/cost, update to latest
        text_gen_config = {"temperature": 0.1} # Lower temp for consistent JSON output

        model = genai.GenerativeModel(model_name=TEXT_MODEL, generation_config=text_gen_config)
        gemini_response = model.generate_content(prompt, request_options={"timeout": 120}) # Increased timeout for Gemini call

        # Check if response has parts and text
        if not gemini_response or not gemini_response.parts:
             st.error("Gemini returned an empty response.")
             return None

        json_text = ''.join(part.text for part in gemini_response.parts).strip()

        # Clean potential markdown formatting (robustly)
        if json_text.startswith("```json"):
            json_text = json_text[7:].strip()
        if json_text.endswith("```"):
            json_text = json_text[:-3].strip()

        st.info("Received response from Gemini. Attempting to parse JSON.")
        parsed_json = json.loads(json_text)
        st.success("JSON parsed successfully.")
        # Check if the structure seems correct (has "menu_data" key which is a list)
        if not isinstance(parsed_json, dict) or "menu_data" not in parsed_json or not isinstance(parsed_json["menu_data"], list):
            st.warning("Gemini returned JSON, but the structure doesn't match the expected format {'menu_data': [...]}.")
            st.text(f"Gemini raw response:\n{json_text[:1000]}...") # Show part of the response
            return None

        return parsed_json

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Gemini response. This usually means Gemini didn't return valid JSON. Details: {e}")
        st.text(f"Gemini raw response (first 1000 chars):\n{json_text[:1000]}...") # Show a part of the raw response
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini extraction: {e}")
        st.text(traceback.format_exc())
        return None

# Pass the API key obtained from UI
def extract_menu_from_multimodal(uploaded_file, api_key):
    st.info(f"Processing {uploaded_file.type} file...")

    try:
        # Read file content as bytes
        file_content_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type

        # Prepare the file part for the Gemini API
        file_part = {
            "mime_type": mime_type,
            "data": file_content_bytes
        }

        # --- Gemini Prompt for Multimodal ---
        prompt_text = """Analyze the content of this document/image which appears to be a restaurant menu. Extract food categories, items, prices, and any associated dietary information (like vegan, vegetarian, Jain) or allergy information (like nut-free, gluten-free).

        Pay close attention to the visual layout and text to identify categories and their items. Associate prices correctly with items. Extract dietary/allergy info only if it appears directly next to or clearly associated with an item.

        For each item, if dietary or allergy information is present, include it as a list of strings for the keys "diet_types" and "allergy_info". If no information is found for an item, the lists should be empty (e.g., "diet_types": [], "allergy_info": []). Price should be extracted as a string including currency symbol if present. If a price is not explicitly listed but implied (e.g., "Ask"), capture that string. If no price is found or it's unclear, use "N/A".

        If you cannot confidently identify a menu structure or items, return an empty JSON object like {{"menu_data": []}}.

        Format the output strictly as a JSON object following this structure:
        {{
            "menu_data": [
                {{
                    "category_name": "Category Name",
                    "items": [
                        {{
                            "item_name": "Item Name",
                            "price": "Price String",
                            "diet_types": ["Diet Type 1", "Diet Type 2"],
                            "allergy_info": ["Allergy Info 1", "Allergy Info 2"]
                        }},
                        // ... other items in this category ...
                    ]
                }},
                // ... other categories ...
            ]
        }}
        Only return the JSON object. Do not include any introductory or concluding text, explanations, or formatting like markdown code blocks (```json ```). Ensure the output is valid JSON parseable by json.loads().
        """

        st.info("Sending file content to Gemini for extraction...")
        # Configure genai using the provided API key *before* making the call
        genai.configure(api_key=api_key) # Use the key from UI

        MULTIMODAL_MODEL = "gemini-1.5-flash-latest" # Use latest version
        multimodal_gen_config = {"temperature": 0.1} # Adjust as needed

        model = genai.GenerativeModel(model_name=MULTIMODAL_MODEL, generation_config=multimodal_gen_config)
        # The content is a list: [text_part, file_part]
        gemini_response = model.generate_content([prompt_text, file_part], request_options={"timeout": 120}) # Increased timeout

        if not gemini_response or not gemini_response.parts:
             st.error("Gemini returned an empty response.")
             return None

        json_text = ''.join(part.text for part in gemini_response.parts).strip()

        # Clean potential markdown formatting (robustly)
        if json_text.startswith("```json"):
            json_text = json_text[7:].strip()
        if json_text.endswith("```"):
            json_text = json_text[:-3].strip()

        st.info("Received response from Gemini. Attempting to parse JSON.")
        parsed_json = json.loads(json_text)
        st.success("JSON parsed successfully.")

        # Check if the structure seems correct
        if not isinstance(parsed_json, dict) or "menu_data" not in parsed_json or not isinstance(parsed_json["menu_data"], list):
            st.warning("Gemini returned JSON, but the structure doesn't match the expected format {'menu_data': [...]}.")
            st.text(f"Gemini raw response:\n{json_text[:1000]}...") # Show part of the response
            return None


        return parsed_json

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Gemini response. This usually means Gemini didn't return valid JSON. Details: {e}")
        st.text(f"Gemini raw response (first 1000 chars):\n{json_text[:1000]}...") # Show a part of the raw response
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during multimodal extraction: {e}")
        st.text(traceback.format_exc())
        return None


# --- Load Second Excel ---
def load_second_excel(uploaded_file):
    st.info("Loading second Excel file...")
    try:
        df = pd.read_excel(uploaded_file)
        # Optional: Validate columns
        required_cols = ["Category", "Item", "Price", "Diet Types", "Allergy Info"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"The uploaded Excel is missing expected columns: {', '.join(missing_cols)}. Comparison might be incomplete.")
            # Attempt to create missing columns with default values
            for col in missing_cols:
                df[col] = "N/A" if col in ["Price", "Category", "Item"] else "" # Use empty string for lists

        # Ensure diet_types and allergy_info columns exist and are string type for comparison
        for col in ["Diet Types", "Allergy Info"]:
            if col not in df.columns:
                 df[col] = "" # Add missing column
            df[col] = df[col].fillna("").astype(str) # Ensure string and fill NaNs

        # Ensure other key columns are string type
        for col in ["Category", "Item", "Price"]:
             if col in df.columns:
                 df[col] = df[col].fillna("N/A").astype(str)


        st.success("Second Excel loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading second Excel file: {e}")
        st.text(traceback.format_exc())
        return None

def convert_to_df(menu_data):
    rows = []
    # Safely iterate through menu_data list, defaulting to empty list if the "menu_data" key is missing
    # or if menu_data is None
    if menu_data is None or not isinstance(menu_data, dict) or "menu_data" not in menu_data or not isinstance(menu_data["menu_data"], list):
        st.warning("No valid menu data provided to convert to DataFrame.")
        return pd.DataFrame(columns=["Category", "Item", "Price", "Diet Types", "Allergy Info"]) # Return empty DF

    for cat in menu_data.get("menu_data", []): # Use .get for safety
        category_name = str(cat.get("category_name", "Uncategorized")).strip()

        for item in cat.get("items", []): # Use .get for safety
            item_name = str(item.get("item_name", "Unknown")).strip()
            price = str(item.get("price", "N/A")).strip()

            diet_types_list = item.get("diet_types", [])
            allergy_info_list = item.get("allergy_info", [])

            # Ensure they are lists before joining, handle non-list cases
            diet_types_str = ", ".join(diet_types_list) if isinstance(diet_types_list, list) else str(diet_types_list or "")
            allergy_info_str = ", ".join(allergy_info_list) if isinstance(allergy_info_list, list) else str(allergy_info_list or "")

            rows.append([category_name, item_name, price, diet_types_str, allergy_info_str])

    columns = ["Category", "Item", "Price", "Diet Types", "Allergy Info"]

    # Create DataFrame, ensure string types to prevent comparison issues
    df = pd.DataFrame(rows, columns=columns)
    for col in columns:
        df[col] = df[col].astype(str)

    return df

# --- Create Download Link ---
def create_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Mismatches')
    output.seek(0)
    return output


# --- Fuzzy Comparison Logic ---
# (Updated to provide more detailed mismatch types)
def compare_menus_fuzzy(df_uploaded, df_extracted, item_name_threshold, category_threshold, very_high_name_threshold):

    mismatches = []
    # Convert extracted data to a list of dicts for easier processing
    extracted_items_list = df_extracted.to_dict(orient='records')
    # Keep track of which extracted items have been considered as a match (above threshold)
    matched_extracted_indices = set()

    # Iterate through each item in the uploaded menu (the reference)
    for uploaded_index, uploaded_row in df_uploaded.iterrows():
        # Safely get values, converting to string and stripping whitespace
        # Ensure columns exist before accessing them
        uploaded_category = str(uploaded_row.get("Category", "N/A")).strip()
        uploaded_item = str(uploaded_row.get("Item", "N/A")).strip()
        uploaded_price = str(uploaded_row.get("Price", "N/A")).strip()
        uploaded_diet = str(uploaded_row.get("Diet Types", "")).strip() # Get as string for comparison
        uploaded_allergy = str(uploaded_row.get("Allergy Info", "")).strip() # Get as string for comparison


        # Skip rows in uploaded data that are missing item name or are clearly just headers/footers/empty
        if not uploaded_item or uploaded_item.lower() in ['n/a', 'item', 'unknown'] or len(uploaded_item) < 3:
             continue

        # Find the best fuzzy match for the uploaded item name among ALL extracted items
        best_score = 0
        best_match_item = None
        best_match_index = -1

        for extracted_index, extracted_item in enumerate(extracted_items_list):
            extracted_item_name = str(extracted_item.get("Item", "N/A")).strip()

            # Skip extracted items missing name or are likely headers/footers/empty
            if not extracted_item_name or extracted_item_name.lower() in ['n/a', 'item', 'unknown'] or len(extracted_item_name) < 3:
                 continue

            # Calculate fuzzy score for item name (case-insensitive)
            score = fuzz.ratio(uploaded_item.lower(), extracted_item_name.lower())

            # Keep track of the best match found so far for *this specific* uploaded item
            if score > best_score:
                best_score = score
                best_match_item = extracted_item
                best_match_index = extracted_index

        # --- Decision Logic based on Best Match Found for the current uploaded item ---

        # If a best match was found above the general threshold
        if best_score >= item_name_threshold and best_match_item is not None:
            # We found a plausible match in extracted data for this uploaded item.
            # Mark the extracted item's index. This item will NOT be reported as an "Extra Item in Extracted" later.
            matched_extracted_indices.add(best_match_index)

            # Get details of the matched extracted item
            matched_category = str(best_match_item.get("Category", "N/A")).strip()
            matched_item_name = str(best_match_item.get("Item", "N/A")).strip() # The actual name from extracted
            matched_price = str(best_match_item.get("Price", "N/A")).strip()
            matched_diet = str(best_match_item.get("Diet Types", "")).strip() # Get as string
            matched_allergy = str(best_match_item.get("Allergy Info", "")).strip() # Get as string

            # Compare details
            is_category_mismatch = fuzz.ratio(uploaded_category.lower(), matched_category.lower()) < category_threshold
            is_price_mismatch = uploaded_price != matched_price
            is_diet_mismatch = uploaded_diet != matched_diet
            is_allergy_mismatch = uploaded_allergy != matched_allergy


            # --- Determine the specific issue based on score and detail mismatches ---
            if best_score >= very_high_name_threshold:
                 # Item names are very similar ("Appetizer" vs "Appetizers") - Likely the same item
                 if is_category_mismatch or is_price_mismatch or is_diet_mismatch or is_allergy_mismatch:
                     # Details differ for this very similar item
                     # Details are implicitly shown in the columns now
                     mismatches.append({
                         "Issue": "Details Mismatch (Very High Name Similarity)",
                         "Uploaded Category": uploaded_category,
                         "Uploaded Item": uploaded_item,
                         "Uploaded Price": uploaded_price,
                         "Uploaded Diet/Allergy": f"Diet: {uploaded_diet}, Allergy: {uploaded_allergy}",
                         "Source Category": matched_category,
                         "Source Item": matched_item_name,
                         "Source Price": matched_price,
                         "Source Diet/Allergy": f"Diet: {matched_diet}, Allergy: {matched_allergy}",
                         "Item Name Score (%)": f"{best_score:.1f}"
                     })
                 # Else: Names are very similar, and all checked details match. No mismatch to report.

            else: # best_score is between item_name_threshold and very_high_name_threshold
                # Item names are similar enough to be considered a potential match with a typo ("Onion" vs "Anion")
                 mismatches.append({
                    "Issue": "Spelling/Name Difference",
                    "Uploaded Category": uploaded_category,
                    "Uploaded Item": uploaded_item,
                    "Uploaded Price": uploaded_price,
                    "Uploaded Diet/Allergy": f"Diet: {uploaded_diet}, Allergy: {uploaded_allergy}",
                    "Source Category": matched_category,
                    "Source Item": matched_item_name,
                    "Source Price": matched_price,
                    "Source Diet/Allergy": f"Diet: {matched_diet}, Allergy: {matched_allergy}",
                    "Item Name Score (%)": f"{best_score:.1f}"
                 })
                # Note: For spelling differences, we primarily flag the name difference. We don't explicitly list
                # Category/Price mismatches here unless specifically required, as the main issue is the name.


        else: # best_score < item_name_threshold
            # No sufficiently similar item found in extracted data
            best_match_item_name_detail = str(best_match_item.get('Item', 'N/A')).strip() if best_match_item else 'None Found'
            best_match_score_detail = f"{best_score:.1f}" if best_match_item else "N/A"

            mismatches.append({
                "Issue": "Not Found in Source",
                "Uploaded Category": uploaded_category,
                "Uploaded Item": uploaded_item,
                "Uploaded Price": uploaded_price,
                "Uploaded Diet/Allergy": f"Diet: {uploaded_diet}, Allergy: {uploaded_allergy}",
                "Source Category": "Not Found",
                "Source Item": "Not Found",
                "Source Price": "-",
                "Source Diet/Allergy": "-",
                "Item Name Score (%)": best_match_score_detail # Show the score of the closest found item
            })


    # --- Identify Extra Items in Extracted Data ---
    # Loop through the original extracted list again to find items that were never marked as matched
    for extracted_index, extracted_item in enumerate(extracted_items_list):
        # If the index was NOT added to our matched set, it's an extra item
        # Add a basic check to avoid listing obvious headers/footers/empty rows from extraction
        extracted_item_name = str(extracted_item.get("Item", "N/A")).strip()
        if extracted_index not in matched_extracted_indices and extracted_item_name and extracted_item_name.lower() not in ['n/a', 'item', 'unknown'] and len(extracted_item_name) >= 3:

             extracted_category = str(extracted_item.get("Category", "N/A")).strip()
             extracted_price = str(extracted_item.get("Price", "N/A")).strip()
             extracted_diet = str(extracted_item.get("Diet Types", "")).strip()
             extracted_allergy = str(extracted_item.get("Allergy Info", "")).strip()

             mismatches.append({
                "Uploaded Category": "-",
                "Uploaded Item": "-",
                "Uploaded Price": "-",
                "Uploaded Diet/Allergy": "-",
                "Source Category": extracted_category,
                "Source Item": extracted_item_name,
                "Source Price": extracted_price,
                "Source Diet/Allergy": f"Diet: {extracted_diet}, Allergy: {extracted_allergy}",
                "Issue": "Extra Item in Source",
                
            })

    # Create the final DataFrame for mismatches
    comparison_df = pd.DataFrame(mismatches)

    # Define the desired column order, including the new 'Details' and Diet/Allergy columns
    # Re-ordering columns for better readability
    column_order = [
        
        "Uploaded Category",
        "Uploaded Item",
        "Uploaded Price",
        "Uploaded Diet/Allergy", # Combined Diet/Allergy
        "Source Category",
        "Source Item",
        "Source Price",
        "Source Diet/Allergy",
        "Issue", # Put issue first
    ]

    # Reindex the DataFrame to ensure the correct order
    # Handle cases where the DataFrame is empty (no mismatches)
    if not comparison_df.empty:
        # Ensure all columns in column_order exist in the DataFrame before reindexing
        # This handles cases where certain mismatch types might not generate all columns
        existing_cols = comparison_df.columns.tolist()
        cols_to_include = [col for col in column_order if col in existing_cols]
        comparison_df = comparison_df.reindex(columns=cols_to_include)

    return comparison_df


# --- Streamlit Layout ---
st.set_page_config(layout="wide", page_title="Menu Comparison Tool")

st.title("Menu Comparison Tool")
# st.markdown("""
# Compare your reference menu (uploaded Excel) against a menu extracted from a Website URL, PDF, Image, or another Excel.
# The tool uses AI to extract data and fuzzy matching to find differences like missing items, spelling errors, and price/category mismatches.
# """)

# --- API Key Input (Moved to the top) ---
st.subheader("Google Gemini API Key (Required for URL, PDF, Image extraction)")
entered_api_key = st.text_input(
    "Enter your Google Gemini API Key:",
    type="password", # Hide the key input
    key="gemini_api_key_input",
    help="Required for extracting menus from Website URLs, PDFs, and Images."
)

# Check if API key is provided (after stripping whitespace)
api_key_ok = bool(entered_api_key and entered_api_key.strip())

# Configure genai ONLY if a valid key is provided
if api_key_ok:
    try:
        genai.configure(api_key=entered_api_key.strip())
        # Optional: Add a small check if configuration was successful (though genai doesn't provide a direct method)
        # We assume if genai.configure doesn't raise an immediate error, it's configured.
        st.success("Gemini API Key configured.")
    except Exception as e:
         st.error(f"Failed to configure Gemini API with the provided key: {e}")
         api_key_ok = False # Mark key as not OK if configuration failed

# Display warning and info if API key is missing/invalid
if not api_key_ok:
    st.warning("API key is not configured or invalid. Extraction from Website URL, PDF, and Image sources is disabled.")
    with st.expander("ℹ️ How to set up Google Gemini API Key"):
        st.write("""
        To access this tool, you'll need to provide a Google Gemini API Key. Here's how to obtain one:

1. Visit the Google AI for Developers website: https://ai.google.dev/
2. Click on the "Explore in Google AI Studio" button
3. You'll be redirected to the API key management page (https://aistudio.google.com/apikey)
4. Click on "Create API Key"
5. Select an existing Google project from the dropdown menu (or create a new one if needed)
6. Click "Create" to generate your API key
7.                                                                                                                                               Copy your newly created API key and paste it into the text box above to enable all extraction features
        """)

st.markdown("---") # Separator below the API key section

# Initialize session state variables (keeping these here is fine)
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
if 'df_extracted' not in st.session_state:
    st.session_state.df_extracted = None
if 'extraction_source' not in st.session_state:
     # Set initial source based on API key status
     st.session_state.extraction_source = "Website URL" if api_key_ok else "Another Excel File"
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame() # Start with empty DF

# Remove old threshold state if it exists
for key in ['item_name_threshold', 'category_threshold', 'very_high_name_threshold']:
    if key in st.session_state:
        del st.session_state[key]


col1, col2 = st.columns([1, 2]) # Adjust column widths

# Column 1: Upload Reference Excel (Source of Truth)
with col1:
    st.header("1. Reference Menu")
    st.write("Upload your reference menu Excel file (Your Source of Truth).")
    uploaded_file_ref = st.file_uploader("Upload the Excel file (.xlsx)", type=["xlsx"], key="upload_ref_excel")

    if uploaded_file_ref is not None:
        try:
            # Reset extracted and comparison data if a new reference is uploaded
            if st.session_state.df_uploaded is None or uploaded_file_ref.name != st.session_state.get('uploaded_ref_name'):
                 st.session_state.df_extracted = None
                 st.session_state.comparison_df = pd.DataFrame()
                 st.session_state.uploaded_ref_name = uploaded_file_ref.name # Store name to detect change

            df_ref = pd.read_excel(uploaded_file_ref)
            # Basic validation/cleaning for reference DF too
            required_cols_ref = ["Category", "Item", "Price", "Diet Types", "Allergy Info"]
            for col in required_cols_ref:
                if col not in df_ref.columns:
                    st.warning(f"Reference Excel missing column: '{col}'. Adding with default values.")
                    df_ref[col] = "N/A" if col in ["Price", "Item"] else "" if col == "Category" else "" # Use empty string for lists
                df_ref[col] = df_ref[col].fillna("").astype(str) # Ensure string and fill NaNs


            st.session_state.df_uploaded = df_ref
            st.success("Reference Excel Uploaded & Processed")
            st.dataframe(st.session_state.df_uploaded.head()) # Show only head for large files
        except Exception as e:
            st.error(f"Error reading reference Excel file: {e}")
            st.session_state.df_uploaded = None
            st.session_state.uploaded_ref_name = None # Reset name
            st.session_state.df_extracted = None # Reset other states on error
            st.session_state.comparison_df = pd.DataFrame()


    else:
         st.session_state.df_uploaded = None # Reset if file is cleared
         st.session_state.uploaded_ref_name = None
         st.session_state.df_extracted = None
         st.session_state.comparison_df = pd.DataFrame()


# Column 2: Select Source, Extract/Load Second Menu, Compare
with col2:
    st.header("2. Menu to Compare & Analysis")

    st.write("Select the source for the menu you want to compare against:")

    # Radio buttons for source selection - Options depend on API key availability
    options = ["Another Excel File"]
    if api_key_ok:
        options = ["Website URL", "PDF File", "Image File"] + options

    # Ensure the currently selected source is still in the available options
    # If not, default to "Another Excel File"
    if st.session_state.extraction_source not in options:
         st.session_state.extraction_source = "Another Excel File"

    st.session_state.extraction_source = st.radio(
        "Source Type:",
        options,
        key="source_selector",
        horizontal=True
    )

    extracted_data = None # Variable to hold raw Gemini JSON or DataFrame for Excel
    uploaded_file_second = None # Variable to hold uploaded file object (for PDF/Image/Excel)

    # Conditional input based on source selection AND API key availability
    # Note: The API key check is now done *implicitly* by whether the option is available in the radio button.
    # We still keep the explicit check here for clarity and safety, although the UI prevents selecting
    # the Gemini options if api_key_ok is False.

    if st.session_state.extraction_source == "Website URL": # This option is only visible if api_key_ok is True
        menu_url = st.text_input("Paste menu URL here:", key="menu_url_input")
        if st.button("Extract from URL", key="extract_url_button"):
            if menu_url and api_key_ok: # Double check api_key_ok before calling
                # Clear previous extracted data when starting a new extraction
                st.session_state.df_extracted = None
                st.session_state.comparison_df = pd.DataFrame()
                with st.spinner(f"Extracting menu from {menu_url} using Gemini..."):
                    extracted_data = scrape_menu_with_gemini_from_url(menu_url, entered_api_key.strip()) # Pass key
                    if extracted_data:
                        st.session_state.df_extracted = convert_to_df(extracted_data) # Store DF in state
                    else:
                         st.error("Extraction failed or returned empty data.")
            elif not menu_url:
                 st.warning("Please enter a URL to extract from.")
            # else: # api_key_ok was False, button shouldn't have been clickable but for safety
            #     st.error("Gemini API Key is not configured. Cannot extract from URL.")

        # Reset extracted state if the input field is cleared AND it was the selected source AND data exists
        if not st.session_state.get("menu_url_input") and st.session_state.df_extracted is not None and st.session_state.extraction_source == "Website URL":
             st.session_state.df_extracted = None
             st.session_state.comparison_df = pd.DataFrame()


    elif st.session_state.extraction_source in ["PDF File", "Image File"]: # These options are only visible if api_key_ok is True
        file_type = "PDF" if st.session_state.extraction_source == "PDF File" else "Image"
        file_types_list = ["pdf"] if file_type == "PDF" else ["png", "jpg", "jpeg", "webp"]
        uploaded_file_second = st.file_uploader(
            f"Upload {file_type} file (.{'/'.join(file_types_list)})",
            type=file_types_list,
            key="upload_multimodal_file"
        )
        if st.button(f"Extract from {file_type}", key=f"extract_{file_type}_button"):
            if uploaded_file_second and api_key_ok: # Double check api_key_ok before calling
                # Clear previous extracted data
                st.session_state.df_extracted = None
                st.session_state.comparison_df = pd.DataFrame()
                with st.spinner(f"Extracting menu from {file_type} using Gemini (multimodal)..."):
                    extracted_data = extract_menu_from_multimodal(uploaded_file_second, entered_api_key.strip()) # Pass key
                    if extracted_data:
                        st.session_state.df_extracted = convert_to_df(extracted_data) # Store DF in state
                    else:
                         st.error("Extraction failed or returned empty data.")
            elif not uploaded_file_second:
                 st.warning(f"Please upload a {file_type} file to extract from.")
            # else: # api_key_ok was False
            #     st.error("Gemini API Key is not configured. Cannot extract from files.")

        # Reset extracted state if the file uploader is cleared AND it was the selected source AND data exists
        if uploaded_file_second is None and st.session_state.df_extracted is not None and st.session_state.extraction_source in ["PDF File", "Image File"]:
             st.session_state.df_extracted = None
             st.session_state.comparison_df = pd.DataFrame()


    elif st.session_state.extraction_source == "Another Excel File": # This option is always visible
        uploaded_file_second = st.file_uploader("Upload the second Excel file (.xlsx)", type=["xlsx"], key="upload_second_excel")
        if st.button("Load Second Excel", key="load_second_excel_button"):
            if uploaded_file_second:
                # Clear previous extracted data
                st.session_state.df_extracted = None
                st.session_state.comparison_df = pd.DataFrame()
                # Loading Excel directly results in a DataFrame
                st.session_state.df_extracted = load_second_excel(uploaded_file_second) # Store DF in state
                if st.session_state.df_extracted is None or st.session_state.df_extracted.empty:
                     st.error("Failed to load data from the second Excel file.")
            else:
                 st.warning("Please upload the second Excel file to load.")
        # Reset extracted state if the file uploader is cleared AND it was the selected source AND data exists
        if uploaded_file_second is None and st.session_state.df_extracted is not None and st.session_state.extraction_source == "Another Excel File":
             st.session_state.df_extracted = None
             st.session_state.comparison_df = pd.DataFrame()


    # Display the extracted/loaded DataFrame if available in session state
    if st.session_state.df_extracted is not None and not st.session_state.df_extracted.empty:
        st.subheader("Preview of Second Menu Data")
        st.dataframe(st.session_state.df_extracted.head()) # Show head
        st.success(f"✅ Second menu data processed from {st.session_state.extraction_source}.")
    elif st.session_state.df_extracted is not None and st.session_state.df_extracted.empty:
         st.warning("Processed data from the second source is empty.")
    else:
         st.info("Second menu data is not yet loaded/extracted. Select a source and process.")


    st.markdown("---") # Separator

    st.header("3. Compare Menus")

    if st.session_state.df_uploaded is not None and st.session_state.df_extracted is not None and not st.session_state.df_extracted.empty:

        # Display the fixed comparison thresholds (Optional, commented out previously)
        # st.subheader("Comparison Settings (Fixed Thresholds)")
        # st.write(f"- Item Name Match Threshold: **{DEFAULT_ITEM_NAME_THRESHOLD}%**")
        # st.write(f"- Very High Name Similarity Threshold: **{DEFAULT_VERY_HIGH_NAME_THRESHOLD}%**")
        # st.write(f"- Category Match Threshold: **{DEFAULT_CATEGORY_THRESHOLD}%**")
        # st.info("These thresholds determine how closely items and categories must match to be considered the same or similar during comparison.")


        if st.button("Run Comparison", key="run_comparison_button"):
            with st.spinner("Comparing menus using fuzzy matching..."):
                comparison_df = compare_menus_fuzzy(
                    st.session_state.df_uploaded,
                    st.session_state.df_extracted,
                    item_name_threshold=DEFAULT_ITEM_NAME_THRESHOLD, # Use default
                    category_threshold=DEFAULT_CATEGORY_THRESHOLD, # Use default
                    very_high_name_threshold=DEFAULT_VERY_HIGH_NAME_THRESHOLD # Use default
                )
                st.session_state.comparison_df = comparison_df # Store comparison result in state
                st.success("Comparison Complete!")

        # Display comparison results if available in session state
        if not st.session_state.comparison_df.empty:
            st.subheader("Comparison Results (Mismatches)")
            st.warning(f"Found {len(st.session_state.comparison_df)} potential mismatches or differences.")
            st.dataframe(st.session_state.comparison_df)

            # Download button for the comparison results
            excel_output = create_download_link(st.session_state.comparison_df)
            st.download_button(
                label="Download Mismatches Excel",
                data=excel_output,
                file_name="menu_mismatches.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_mismatches_button"
            )
        elif 'comparison_df' in st.session_state and st.session_state.comparison_df.empty and st.session_state.df_uploaded is not None and st.session_state.df_extracted is not None and not st.session_state.df_extracted.empty:
            # Only show success if comparison ran and found no mismatches
            st.success("🎉 No significant mismatches found between the two menus based on current settings!")

    else:
        if st.session_state.df_uploaded is None:
             st.info("Please upload your reference Excel menu (Step 1).")
        elif st.session_state.df_extracted is None or st.session_state.df_extracted.empty:
             st.info("Please select a source and successfully process the second menu data (Step 2) to enable comparison.")
