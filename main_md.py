import json
import re
from datetime import datetime
from typing import List, Dict, Any


def parse_date(date_str: str) -> tuple:
    """
    Parse date string and return (datetime_object, formatted_string)
    Handles various date formats from Zotero
    """
    if not date_str:
        return datetime.min, "Unknown"

    # Remove any extra whitespace
    date_str = date_str.strip()

    # Try different date formats
    date_formats = [
        "%Y/%m/%d",  # 2015/07/01
        "%Y-%m-%d",  # 2015-07-01
        "%Y/%m",  # 2015/07
        "%Y-%m",  # 2015-07
        "%Y",  # 1990
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Format the output based on the precision
            if fmt == "%Y":
                return dt, date_str
            elif fmt in ["%Y/%m", "%Y-%m"]:
                return dt, dt.strftime("%B %Y")
            else:
                return dt, dt.strftime("%B %d, %Y")
        except ValueError:
            continue

    # If no format matches, return as is
    return datetime.min, date_str


def format_authors(creators: List[Dict]) -> str:
    """Format author list from Zotero creators"""
    if not creators:
        return "Unknown"

    authors = []
    for creator in creators:
        if creator.get("creatorType") == "author":
            first_name = creator.get("firstName", "")
            last_name = creator.get("lastName", "")
            if first_name and last_name:
                authors.append(f"{first_name} {last_name}")
            elif last_name:
                authors.append(last_name)

    if not authors:
        return "Unknown"
    elif len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]}, {authors[1]}"
    else:
        return f"{', '.join(authors[:-1])}, {authors[-1]}"


def format_tags(tags: List[Dict]) -> str:
    """Format tags from Zotero"""
    if not tags:
        return "None"

    tag_names = [tag.get("tag", "") for tag in tags if tag.get("tag")]
    return ", ".join(tag_names) if tag_names else "None"


def clean_abstract(abstract: str) -> str:
    """Clean abstract text by removing HTML entities and extra whitespace"""
    if not abstract:
        return "No abstract available"

    # Replace common HTML entities
    abstract = abstract.replace('â€"', "—")
    abstract = abstract.replace("â€™", "'")
    abstract = abstract.replace("â€œ", '"')
    abstract = abstract.replace("â€", '"')

    # Remove extra whitespace
    abstract = re.sub(r"\s+", " ", abstract.strip())

    return abstract


def process_zotero_json(json_file_path: str, output_file_path: str = None) -> str:
    """
    Process Zotero JSON export file and generate Markdown

    Args:
        json_file_path: Path to the Zotero JSON file
        output_file_path: Optional path to save the output Markdown file

    Returns:
        Generated Markdown content as string
    """

    # Load JSON data
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_file_path}")

    # Extract items
    items = data.get("items", [])

    # Filter journal articles and sort by date (newest first)
    papers = []
    for item in items:
        if item.get("itemType") == "journalArticle":
            date_obj, date_str = parse_date(item.get("date", ""))
            papers.append({"item": item, "date_obj": date_obj, "date_str": date_str})

    # Sort by date (newest first)
    papers.sort(key=lambda x: x["date_obj"], reverse=True)

    # Generate Markdown content
    # <img src="image.jpg" alt="描述" width="300">
    markdown_lines = ['<img src="./assets/matverse_logo.png" alt="" width="300">\n', "# MatVerse AI4(M)S Paper Collection\n",
                      "This is a regularly updated paper collection about AI for science, with a specific focus on materials science, "
                      "associated with the MatVerse paper.\n",
                      "---\n"]
    num_papers = len(papers)
    for i, paper in enumerate(papers, 1):
        item = paper["item"]

        # Basic information
        title = item.get("title", "Untitled")
        authors = format_authors(item.get("creators", []))
        venue = item.get("publicationTitle", "Unknown")
        date_str = paper["date_str"]

        # Volume and issue information
        volume = item.get("volume", "")
        issue = item.get("issue", "")
        pages = item.get("pages", "")
        doi = item.get("DOI", "")

        # Abstract and tags
        abstract = clean_abstract(item.get("abstractNote", ""))
        tags = format_tags(item.get("tags", []))

        # Build paper section
        markdown_lines.append(f"## {num_papers - i + 1}. {title}\n")
        markdown_lines.append(f"**Authors:** {authors}\n")
        markdown_lines.append(f"**Venue:** {venue}\n")
        markdown_lines.append(f"**Publication Date:** {date_str}\n")

        # Add volume/issue/pages if available
        vol_info = []
        if volume:
            vol_info.append(f"Volume {volume}")
        if issue:
            vol_info.append(f"Issue {issue}")

        if vol_info:
            markdown_lines.append(f"**Volume & Issue:** {', '.join(vol_info)}\n")

        if pages:
            markdown_lines.append(f"**Pages:** {pages}\n")

        if doi:
            markdown_lines.append(f"**DOI:** {doi}\n")

        markdown_lines.append(f"**Abstract:**\n{abstract}\n")
        markdown_lines.append(f"**Tags:** {tags}\n")

        # Add separator except for the last item
        if i < len(papers):
            markdown_lines.append("---\n")

    # Add summary
    markdown_lines.append("---\n")
    current_time = datetime.now()
    markdown_lines.append(f"\n**Total Papers:** {len(papers)}\n")
    markdown_lines.append(f"**Generation Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}",)

    # Join all lines
    markdown_content = "\n".join(markdown_lines)

    # Save to file if output path is provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown file saved to: {output_file_path}")
        print(f"\nTotal Papers: {len(papers)}\n")

    return markdown_content


def main():
    """
    Main function to demonstrate usage
    """
    # Example usage
    input_file = "AI4S.json"  # Replace with your JSON file path
    output_file = "README.md"  # Output markdown file

    try:
        process_zotero_json(input_file, output_file)
        print("Processing completed successfully!")
        # print(f"\nFirst 500 characters of generated markdown:")
        # print(
        #     markdown_content[:500] + "..."
        #     if len(markdown_content) > 500
        #     else markdown_content
        # )

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()


# Additional utility functions


def batch_process_zotero_files(input_directory: str, output_directory: str = None):
    """
    Process multiple Zotero JSON files in a directory

    Args:
        input_directory: Directory containing JSON files
        output_directory: Directory to save markdown files (optional)
    """
    import os

    if output_directory is None:
        output_directory = input_directory

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process all JSON files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            input_path = os.path.join(input_directory, filename)
            output_filename = filename.replace(".json", "_papers.md")
            output_path = os.path.join(output_directory, output_filename)

            try:
                process_zotero_json(input_path, output_path)
                print(f"Processed: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def generate_summary_stats(json_file_path: str) -> Dict[str, Any]:
    """
    Generate summary statistics from Zotero collection

    Args:
        json_file_path: Path to the Zotero JSON file

    Returns:
        Dictionary containing summary statistics
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    journal_articles = [
        item for item in items if item.get("itemType") == "journalArticle"
    ]

    # Count by publication year
    year_counts = {}
    venue_counts = {}

    for item in journal_articles:
        # Extract year
        date_str = item.get("date", "")
        if date_str:
            year = date_str.split("/")[0] if "/" in date_str else date_str.split("-")[0]
            year_counts[year] = year_counts.get(year, 0) + 1

        # Count venues
        venue = item.get("publicationTitle", "Unknown")
        venue_counts[venue] = venue_counts.get(venue, 0) + 1

    return {
        "total_items": len(items),
        "journal_articles": len(journal_articles),
        "year_distribution": dict(sorted(year_counts.items(), reverse=True)),
        "venue_distribution": dict(
            sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "top_venues": list(
            sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
        )[:10],
    }
