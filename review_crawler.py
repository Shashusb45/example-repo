import requests
from bs4 import BeautifulSoup
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to fetch reviews from the dummy eCommerce product page
def fetch_reviews_from_dummy_website(url):
    """Fetch reviews from the dummy eCommerce website"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the product title (optional)
        title = soup.find('h1', class_='product-title').text.strip()

        # Extract the reviews (assuming reviews are in div with class 'review')
        reviews = []
        review_elements = soup.find_all('div', class_='review')
        for review in review_elements:
            reviews.append(review.text.strip())
        
        logger.info(f"Successfully fetched {len(reviews)} reviews from {title}.")
        return reviews
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return []

# Function to save reviews to a CSV file
def save_reviews_to_csv(reviews, filename='reviews1.csv'):
    """Save reviews to a CSV file"""
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['review'])  # Write the header
            for review in reviews:
                writer.writerow([review])  # Write each review in a new row
        logger.info(f"Successfully saved {len(reviews)} reviews to {filename}.")
    except Exception as e:
        logger.error(f"Error while writing to CSV: {e}")

# Main function
if __name__ == '__main__':
    url = "http://localhost:5001/dummy_ecommerce_website"  # Adjust URL based on where the app is hosted
    reviews = fetch_reviews_from_dummy_website(url)
    if reviews:
        for idx, review in enumerate(reviews, 1):
            print(f"Review {idx}: {review}")
        save_reviews_to_csv(reviews)
    else:
        print("No reviews found.")
