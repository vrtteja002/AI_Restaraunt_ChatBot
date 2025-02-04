import logging
import os
from typing import List, Dict, Optional
from googlemaps import Client
from langchain.schema import Document

LOGGER = logging.getLogger(__name__)

class GooglePlacesAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_PLACES_API_KEY')
        if not self.api_key:
            raise ValueError("Google Places API key not found")
        self.client = Client(key=self.api_key)

    def search_restaurants(self, location: str, radius: int = 5000, max_results: int = 50) -> List[Dict]:
        try:
            results = []
            page_token = None

            while len(results) < max_results:
                response = self.client.places_nearby(
                    location=self._get_location(location),
                    radius=radius,
                    type='restaurant',
                    page_token=page_token
                )

                if not response.get('results'):
                    break

                results.extend(response['results'])
                page_token = response.get('next_page_token')

                if not page_token:
                    break

            return results[:max_results]
        except Exception as e:
            LOGGER.error(f"Error searching restaurants: {str(e)}")
            return []

    def get_place_details(self, place_id: str) -> Optional[Dict]:
        try:
            result = self.client.place(
                place_id=place_id,
                fields=['name', 'formatted_address', 'rating', 'price_level', 
                        'reviews', 'website', 'formatted_phone_number', 'opening_hours']
            )
            return result.get('result')
        except Exception as e:
            LOGGER.error(f"Error fetching place details: {str(e)}")
            return None

    def _get_location(self, location: str) -> Dict:
        try:
            geocode_result = self.client.geocode(location)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                return location
            raise ValueError(f"Could not geocode location: {location}")
        except Exception as e:
            LOGGER.error(f"Geocoding error: {str(e)}")
            raise

    def convert_to_documents(self, places: List[Dict]) -> List[Document]:
        documents = []
        for place in places:
            try:
                details = self.get_place_details(place['place_id'])
                if details:
                    content = (
                        f"Restaurant: {details.get('name', 'N/A')}\n"
                        f"Address: {details.get('formatted_address', 'N/A')}\n"
                        f"Rating: {details.get('rating', 'N/A')}\n"
                        f"Price Level: {'$' * details.get('price_level', 0) or 'N/A'}\n"
                        f"Phone: {details.get('formatted_phone_number', 'N/A')}\n"
                        f"Website: {details.get('website', 'N/A')}\n"
                    )

                    if details.get('reviews'):
                        content += "\nReviews:\n"
                        for review in details['reviews'][:3]:
                            content += f"- Rating: {review.get('rating')}/5\n"
                            content += f"  {review.get('text', '')[:200]}...\n"

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'source': 'google_places',
                            'place_id': place['place_id'],
                            'name': details.get('name'),
                            'rating': details.get('rating'),
                            'price_level': details.get('price_level')
                        }
                    ))
            except Exception as e:
                LOGGER.error(f"Error converting place to document: {str(e)}")
                continue

        return documents
    