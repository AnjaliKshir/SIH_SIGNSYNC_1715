from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB URL from environment variables
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise Exception("MONGO_URL not found in .env file")

# MongoDB connection
client = MongoClient(mongo_url)
db = client["ISL_vdoDB"]
collection = db["videos"]
fs = GridFS(db)

# Store video in MongoDB
def store_video(word, video_path):
    """
    Store a video in MongoDB with a corresponding word as metadata.
    word: The word associated with the video
    video_path: Path to the video file to be stored
    """
    # Check if the video is already in the database
    if fs.exists({"word": word}):
        print(f"Video for '{word}' already exists.")
        return

    # Read video file as binary
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=os.path.basename(video_path), word=word)
        print(f"Stored video for '{word}' with ID: {video_id}")

#Bulk Upload for All Videos
def bulk_store_videos(folder_path):
    """
    Store all videos in a folder to MongoDB with filenames as corresponding words.
    folder_path: Path to the folder containing video files
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):  # Add other video formats if needed
            word = os.path.splitext(filename)[0]  # Use filename without extension as the word
            video_path = os.path.join(folder_path, filename)
            store_video(word, video_path)


if __name__ == "__main__":
    bulk_store_videos("static") 