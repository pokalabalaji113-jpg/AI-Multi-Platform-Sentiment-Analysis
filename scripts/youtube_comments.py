from youtube_comment_downloader import YoutubeCommentDownloader
import re

downloader = YoutubeCommentDownloader()

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_comments(url, limit=200):

    video_id = extract_video_id(url)

    if not video_id:
        return []

    comments = []

    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):

        comments.append(comment["text"])

        if len(comments) >= limit:
            break

    return comments