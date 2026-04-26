import io
from PIL import Image
from google import genai
from config import Config

client = genai.Client(api_key=Config.GEMINI_API_KEY)

class ImageProcessor:
    MIN_WIDTH = 200
    MIN_HEIGHT = 150

    def is_relevant_image(self, image_bytes: bytes) -> bool:
        """Fast pre-filter before sending to Gemini. Skips logos and icons."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

            if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                return False

            aspect_ratio = width / height
            if aspect_ratio < 0.8 or aspect_ratio > 8:
                return False

            return True
        except Exception:
            return False

    def describe_image(self, image_bytes: bytes, context: str = "") -> str | None:
        """
        Send image to Gemini Vision.
        Returns text description if financial, None if irrelevant.
        """
        if not self.is_relevant_image(image_bytes):
            return None

        try:
            img = Image.open(io.BytesIO(image_bytes))
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = (
                "Analyze this image from a financial document.\n\n"
                "First, determine if this is a financial chart, graph, or data "
                "visualization (bar chart, line chart, pie chart, table, etc.) "
                "or something else (logo, decorative image, photo, icon).\n\n"
                "If it IS a financial chart or graph:\n"
                "- Describe exactly what data it shows\n"
                "- Extract all visible numbers, percentages, labels\n"
                "- Note the time periods shown\n"
                "- Describe the trend (increasing, decreasing, stable)\n"
                "- Format: 'FINANCIAL CHART: [your description with all numbers]'\n\n"
                "If it is NOT a financial chart:\n"
                "- Reply only with: NOT_FINANCIAL\n\n"
                f"Context from surrounding document: {context}"
            )

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[prompt, img],
            )
            text = response.text.strip()

            if "NOT_FINANCIAL" in text:
                return None

            return text

        except Exception as e:
            print(f"Image description failed: {e}")
            return None

    def process_images(
        self,
        images: list[dict],
        surrounding_text: str = "",
    ) -> list[dict]:
        """
        Process all images from a document.
        Returns only financial images with descriptions.
        """
        processed = []

        for img in images:
            description = self.describe_image(
                img["bytes"],
                context=surrounding_text[:500],
            )

            if description:
                processed.append({
                    "page": img.get("page"),
                    "description": description,
                    "type": "chart_or_graph",
                })
                print(f"  📊 Financial image found on page {img.get('page')}")
            else:
                print(f"  🚫 Skipped non-financial image on page {img.get('page')}")

        return processed
