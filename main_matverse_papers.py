import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter


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


def generate_yearly_publication_chart(
        papers: List[Dict], output_dir: str = "./assets"
) -> str:
    """
    Generate a stacked bar chart showing yearly publication distribution by journal

    Args:
        papers: List of paper dictionaries with date information
        output_dir: Directory to save the chart image

    Returns:
        Relative path to the generated chart image
    """
    # Create assets directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract years and venues from papers
    yearly_journal_data = {}
    all_journals = set()

    for paper in papers:
        date_obj = paper["date_obj"]
        if date_obj != datetime.min:
            year = date_obj.year
            venue = paper["item"].get("publicationTitle", "Unknown")

            if year not in yearly_journal_data:
                yearly_journal_data[year] = {}

            if venue not in yearly_journal_data[year]:
                yearly_journal_data[year][venue] = 0

            yearly_journal_data[year][venue] += 1
            all_journals.add(venue)

    if not yearly_journal_data:
        return ""

    # Prepare data for plotting
    sorted_years = sorted(yearly_journal_data.keys())

    # Get top journals by total publication count
    journal_totals = {}
    for year_data in yearly_journal_data.values():
        for journal, count in year_data.items():
            journal_totals[journal] = journal_totals.get(journal, 0) + count

    # Sort journals by total count and keep only top journals, group others
    sorted_journals = sorted(
        journal_totals.keys(), key=lambda x: journal_totals[x], reverse=True
    )

    # Define top N journals to show individually (others will be grouped as "Others")
    top_n = 8
    top_journals = sorted_journals[:top_n]
    other_journals = sorted_journals[top_n:]

    # Create a curated color palette with good contrast
    colors = [
        "#2E86AB",  # Blue
        "#A23B72",  # Purple
        "#F18F01",  # Orange
        "#C73E1D",  # Red
        "#4CAF50",  # Green
        "#9C27B0",  # Deep Purple
        "#FF9800",  # Amber
        "#607D8B",  # Blue Grey
        "#8BC34A",  # Light Green (for Others)
    ]

    # Assign colors to journals
    journal_colors = {}
    for i, journal in enumerate(top_journals):
        journal_colors[journal] = colors[i % len(colors)]
    journal_colors["Others"] = colors[-1]  # Grey for others

    # Reorganize data to include "Others" category
    display_journals = top_journals + ["Others"]

    # Prepare data for stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bars
    bottom_values = [0] * len(sorted_years)

    for journal in display_journals:
        journal_counts = []
        for year in sorted_years:
            if journal == "Others":
                # Sum up all other journals
                count = sum(yearly_journal_data[year].get(j, 0) for j in other_journals)
            else:
                count = yearly_journal_data[year].get(journal, 0)
            journal_counts.append(count)

        # Only plot if journal has publications
        if sum(journal_counts) > 0:
            # Shorten journal names for better display
            display_name = journal
            # if len(journal) > 25:
            #     display_name = journal[:22] + "..."

            bars = ax.bar(
                sorted_years,
                journal_counts,
                bottom=bottom_values,
                label=display_name,
                color=journal_colors[journal],
                alpha=0.85,
                edgecolor="white",
                linewidth=1,
            )

            # Update bottom values for next stack
            bottom_values = [b + c for b, c in zip(bottom_values, journal_counts)]

            # Add labels on segments (only if count >= 3 and segment height >= 8% of total)
            for i, (bar, count) in enumerate(zip(bars, journal_counts)):
                if count >= 3:
                    total_height = bottom_values[i]
                    segment_ratio = count / total_height if total_height > 0 else 0

                    if (
                            segment_ratio >= 0.08
                    ):  # Only label if segment is at least 8% of total
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            str(count),
                            ha="center",
                            va="center",
                            fontweight="bold",
                            fontsize=12,
                            color="white",
                            # bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                        )

    # Add total count labels on top of each bar
    for i, (year, total) in enumerate(zip(sorted_years, bottom_values)):
        if total > 0:
            ax.text(
                year,
                total + max(bottom_values) * 0.02,
                str(int(total)),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=18,
                color="black",
                # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

    # Customize the chart
    ax.set_xlabel("Year", fontsize=18, fontweight="bold")
    ax.set_ylabel("Number of Publications", fontsize=18, fontweight="bold")
    ax.set_title(
        "Year and Journal over {} Publications".format(len(papers)), fontsize=18, fontweight="bold", pad=25
    )

    # Add subtle grid
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Customize x-axis
    ax.set_xticks(sorted_years)
    if len(sorted_years) > 10:
        ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="both", which="major", labelsize=18)

    # Style the axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 1),
        framealpha=0.95,
        fontsize=18,
        title="Journals",
        # title_fontweight='bold',
        title_fontsize=18,
        borderaxespad=0,
        columnspacing=1,
        handletextpad=0.5,
    )
    legend.get_title().set_fontweight("bold")

    # ax.legend.get_title().set_fontweight('bold')

    # Adjust layout and save

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "yearly_publications.svg")
    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        # transparent=True
    )
    plt.close()

    # Return relative path for markdown
    return chart_path


def generate_journal_index(papers: List[Dict]) -> str:
    """
    Generate journal index section for the markdown file

    Args:
        papers: List of paper dictionaries with item, date_obj, date_str

    Returns:
        Markdown string for journal index
    """
    # Group papers by journal
    journal_papers = defaultdict(list)

    for i, paper in enumerate(papers):
        item = paper["item"]
        venue = item.get("publicationTitle", "Unknown Journal")
        title = item.get("title", "Untitled")
        paper_number = len(papers) - i  # Since papers are sorted newest first

        journal_papers[venue].append(
            {"title": title, "number": paper_number, "date_str": paper["date_str"]}
        )

    # Sort journals alphabetically
    sorted_journals = sorted(journal_papers.keys())

    # Generate markdown for journal index
    index_lines = [
        "## 📚 Journal Index\n",
        "This section provides a quick overview of papers organized by publication venue for easy navigation.\n",
    ]

    for journal in sorted_journals:
        papers_in_journal = journal_papers[journal]
        # Sort papers within each journal by number (newest first)
        papers_in_journal.sort(key=lambda x: x["number"], reverse=True)

        index_lines.append(f"### {journal} ({len(papers_in_journal)} papers)\n")

        for paper in papers_in_journal:
            # Create anchor link to the paper section
            anchor = f"#{paper['number']}-{paper['title'].lower().replace(' ', '-').replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('[', '').replace(']', '')}"
            index_lines.append(
                f"- [{paper['number']}. {paper['title']}]({anchor}) *({paper['date_str']})*"
            )

        index_lines.append("")  # Add empty line between journals

    index_lines.append("---\n")

    return "\n".join(index_lines)


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

    print("number of items {}".format(len(items)))

    # Filter journal articles and sort by date (newest first)
    papers = []
    for item in items:
        # if item.get("itemType") == "journalArticle":
        date_obj, date_str = parse_date(item.get("date", ""))
        papers.append({"item": item, "date_obj": date_obj, "date_str": date_str})

    # Sort by date (newest first)
    papers.sort(key=lambda x: x["date_obj"], reverse=True)

    # Generate yearly publication chart
    chart_path = generate_yearly_publication_chart(papers)

    # Generate Markdown content

    markdown_lines = [
        f"""<div align="center">
<img src="./assets/matverse_logo.png" width="300"><h1>MatVerse AI4(M)S Paper Collection</h1>
</div>\n\n---\n""",
        "This is a regularly updated paper collection about AI for science, with a specific focus on materials science, "
        "associated with the MatVerse paper.\n",
    ]

    # markdown_lines = [
    #     '<img src="./assets/matverse_logo.png" alt="" width="300">\n',
    #     "# MatVerse AI4(M)S Paper Collection\n",
    #     "This is a regularly updated paper collection about AI for science, with a specific focus on materials science, "
    #     "associated with the MatVerse paper.\n",
    # ]

    # Add yearly publication chart if generated
    if chart_path:
        markdown_lines.extend(
            [
                "## 📈 Publication Timeline\n",
                f'<img src="{chart_path}" alt="Yearly Publication Distribution" width="800">\n',
            ]
        )

    markdown_lines.extend(["---\n"])

    # Add journal index section
    journal_index = generate_journal_index(papers)
    markdown_lines.append(journal_index)

    markdown_lines.append("## 📑 Papers (Chronological Order)\n")

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
        # url = item.get('url', "")

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
            markdown_lines.append(f"**DOI:** https://doi.org/{doi}\n")

        # if url:
        #     markdown_lines.append(f"**URL:** {url}\n")

        markdown_lines.append(f"**Abstract:**\n{abstract}\n")
        markdown_lines.append(f"**Tags:** {tags}\n")

        # Add separator except for the last item
        if i < len(papers):
            markdown_lines.append("---\n")

    # Add summary
    markdown_lines.append("---\n")
    current_time = datetime.now()
    markdown_lines.append(f"\n**Total Papers:** {len(papers)}\n")
    markdown_lines.append(
        f"**Generation Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Join all lines
    markdown_content = "\n".join(markdown_lines)

    # Save to file if output path is provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown file saved to: {output_file_path}")
        if chart_path:
            print(f"Chart saved to: {chart_path}")
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

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()

# Additional utility functions


# def batch_process_zotero_files(input_directory: str, output_directory: str = None):
#     """
#     Process multiple Zotero JSON files in a directory
#
#     Args:
#         input_directory: Directory containing JSON files
#         output_directory: Directory to save markdown files (optional)
#     """
#     import os
#
#     if output_directory is None:
#         output_directory = input_directory
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)
#
#     # Process all JSON files in the directory
#     for filename in os.listdir(input_directory):
#         if filename.endswith(".json"):
#             input_path = os.path.join(input_directory, filename)
#             output_filename = filename.replace(".json", "_papers.md")
#             output_path = os.path.join(output_directory, output_filename)
#
#             try:
#                 process_zotero_json(input_path, output_path)
#                 print(f"Processed: {filename} -> {output_filename}")
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
#
#
# def generate_summary_stats(json_file_path: str) -> Dict[str, Any]:
#     """
#     Generate summary statistics from Zotero collection
#
#     Args:
#         json_file_path: Path to the Zotero JSON file
#
#     Returns:
#         Dictionary containing summary statistics
#     """
#     with open(json_file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     items = data.get("items", [])
#     journal_articles = [
#         item for item in items if item.get("itemType") == "journalArticle"
#     ]
#
#     # Count by publication year
#     year_counts = {}
#     venue_counts = {}
#
#     for item in journal_articles:
#         # Extract year
#         date_str = item.get("date", "")
#         if date_str:
#             year = date_str.split("/")[0] if "/" in date_str else date_str.split("-")[0]
#             year_counts[year] = year_counts.get(year, 0) + 1
#
#         # Count venues
#         venue = item.get("publicationTitle", "Unknown")
#         venue_counts[venue] = venue_counts.get(venue, 0) + 1
#
#     return {
#         "total_items": len(items),
#         "journal_articles": len(journal_articles),
#         "year_distribution": dict(sorted(year_counts.items(), reverse=True)),
#         "venue_distribution": dict(
#             sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
#         ),
#         "top_venues": list(
#             sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
#         )[:10],
#     }
