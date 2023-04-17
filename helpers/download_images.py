import json
import os
import requests
from pprint import pprint

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = os.environ["BRING_SEARCH_KEY"]
endpoint = "https://api.bing.microsoft.com/v7.0/images/search"

# Search terms to query
search_terms = ['gun', 'rifle', 'firegun', 'knife', 'person with cellphone', 'person with gun']

# Number of images to download for each search term
num_images_per_term = 1000

# Loop over each search term and download images
for term in search_terms:
    print(f"Downloading images for '{term}'...")
    
    # Create folder to save images
    folder = f"images/{term}"
    os.makedirs(folder, exist_ok=True)
    
    # Set initial offset and counter
    offset = 0
    num_downloaded = 0
    number_of_calls = 23
    
    # Continue making requests until we have downloaded the desired number of images
    while num_downloaded < num_images_per_term:
        
        # Construct request parameters
        params = {'q': term, 'offset': offset, 'count': 150}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        
        # Make API request
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
        except Exception as ex:
            print(f"Error making request: {ex}")
            break

        number_of_calls += 1
        print(f"number_of_calls = {number_of_calls}")
        
        # Parse response JSON
        try:
            response_json = response.json()
            value = response_json.get('value', [])
            next_offset = response_json.get('nextOffset', None)
        except Exception as ex:
            print(f"Error parsing response JSON: {ex}")
            break
        
        # Download each image in the response
        for img in value:
            try:
                # Get image URL and file extension
                img_url = img['contentUrl']
                ext = os.path.splitext(img_url)[-1]
                
                # Download image and save to file
                filename = f"{folder}/{term}_{num_downloaded}{ext}"
                img_data = requests.get(img_url, timeout=10).content
                with open(filename, 'wb') as f:
                    f.write(img_data)
                
                num_downloaded += 1
                print(f"Downloaded {num_downloaded} images for {term}")
            except Exception as ex:
                print(f"Error downloading image: {ex}")
                continue
        
        # Update offset for next request
        offset = next_offset
        
        # If there are no more results, break out of the loop
        if next_offset is None:
            break
        
    print(f"Downloaded {num_downloaded} images for '{term}'.")
