#this is the image_analysys.py file page

from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    global cv_client
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Analyze image
        AnalyzeImage(image_file, image_data, cv_client)

        # Background removal
        BackgroundForeground(ai_endpoint, ai_key, image_file)

    except Exception as ex:
        print(ex)

def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')
    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ],
        )
    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

    # Display analysis results
    if result.caption:
        print("\nCaption:")
        print(f" Caption: '{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)")

    if result.dense_captions:
        print("\nDense Captions:")
        for caption in result.dense_captions.list:
            print(f" Caption: '{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")

    if result.tags:
        print("\nTags:")
        for tag in result.tags.list:
            print(f" Tag: '{tag.name}' (confidence: {tag.confidence * 100:.2f}%)")

    if result.objects:
        print("\nObjects in image:")
        image = Image.open(image_filename)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        for detected_object in result.objects.list:
            print(f" {detected_object.tags[0].name} (confidence: {detected_object.tags[0].confidence * 100:.2f}%)")
            r = detected_object.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)
            plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

        plt.imshow(image)
        plt.tight_layout(pad=0)
        fig.savefig('objects.jpg')
        print(' Results saved in objects.jpg')

    if result.people:
        print("\nPeople in image:")
        image = Image.open(image_filename)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        for detected_person in result.people.list:
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

        plt.imshow(image)
        plt.tight_layout(pad=0)
        fig.savefig('people.jpg')
        print(' Results saved in people.jpg')

def BackgroundForeground(endpoint, key, image_file):
    api_version = "2023-02-01-preview"
    mode = "foregroundMatting"  # Can be "foregroundMatting" or "backgroundRemoval"

    print('\nRemoving background from image...')
    url = f"{endpoint}computervision/imageanalysis:segment?api-version={api_version}&mode={mode}"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json"
    }
    image_url = f"https://github.com/MicrosoftLearning/mslearn-ai-vision/blob/main/Labfiles/01-analyze-images/Python/image-analysis/{image_file}?raw=true"
    body = {"url": image_url}
    response = requests.post(url, headers=headers, json=body)
    
    with open("background.png", "wb") as file:
        file.write(response.content)
    print(' Results saved in background.png\n')

if __name__ == "__main__":
    main()

