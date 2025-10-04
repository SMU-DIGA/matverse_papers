# import json
# import re
# import os
from datetime import datetime

# from typing import List, Dict, Any
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from collections import Counter
#
conference_list = [
    "Neural Information Processing Systems",
    "International Conference on Machine Learning",
    "International Conference on Learning Representations",
]


def get_venue(paper):
    publication_type = paper["item"].get("itemType", "")

    if publication_type == "journalArticle":
        venue = paper["item"].get("publicationTitle", "")
    elif publication_type == "preprint":
        venue = "Preprint"
    elif publication_type == "conferencePaper":
        conf_name = paper["item"].get("conferenceName", "")
        for conf in conference_list:
            if conf.lower() in conf_name.lower():
                venue = conf
                break
        else:
            venue = conf_name
    else:
        venue = "Unknown"

    return venue


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
