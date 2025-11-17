import json

def restore_placeholders(text, guide_images):
    """
    Replace placeholder links like IMAGE_1 in `text` with their actual URL from `guide_images` in metadata.
    """
    if isinstance(guide_images, str):
        try:
            guide_images = json.loads(guide_images)
        except json.JSONDecodeError:
            return text

    for placeholder, actual_url in guide_images.items():
        text = text.replace(placeholder, actual_url)
    return text


def format_youtube_link(doc) -> str:
    """
    Return a styled markdown hyperlink for YouTube tutorial if the metadata contains a valid link.
    """
    link = doc.metadata.get("youtube", "")
    if not link:
        return ""
    return f"[YouTube Tutorial]({link})"
