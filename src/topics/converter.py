"""
Topic Format Converter
--------------------
Convert between different topic formats (JSON, CSV, XML).
"""

import json
import csv
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def convert_to_nist_format(input_file: str,
                           output_file: str,
                           format: str = 'json') -> int:
    """
    Convert topics to NIST-style XML format.

    Args:
        input_file: Path to input topics file
        output_file: Path to output XML file
        format: Input format ('json', 'csv', 'txt')

    Returns:
        Number of topics converted
    """
    topics = []

    try:
        if format == 'json':
            topics = load_json_topics(input_file)
        elif format == 'csv':
            topics = load_csv_topics(input_file)
        elif format == 'txt':
            topics = load_txt_topics(input_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Convert to XML
        root = ET.Element('topics')

        for topic in topics:
            topic_elem = ET.SubElement(root, 'topic', number=str(topic['number']))

            title = ET.SubElement(topic_elem, 'title')
            title.text = topic['title']

            desc = ET.SubElement(topic_elem, 'description')
            desc.text = topic.get('description', '')

            narr = ET.SubElement(topic_elem, 'narrative')
            narr.text = topic.get('narrative', '')

        # Create pretty XML
        xml_str = ET.tostring(root, 'utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        logger.info(f"Converted {len(topics)} topics to NIST format")
        return len(topics)

    except Exception as e:
        logger.error(f"Error converting topics: {e}")
        raise


def load_json_topics(file_path: str) -> List[Dict]:
    """Load topics from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif 'topics' in data:
        return data['topics']
    else:
        return [data]


def load_csv_topics(file_path: str) -> List[Dict]:
    """Load topics from CSV file"""
    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = {
                'number': row.get('id', row.get('number', '')),
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'narrative': row.get('narrative', '')
            }
            topics.append(topic)
    return topics


def load_txt_topics(file_path: str) -> List[Dict]:
    """Load topics from text file"""
    topics = []
    current_topic = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('Number:'):
            if current_topic:
                topics.append(current_topic)
            current_topic = {'number': line.replace('Number:', '').strip()}
        elif line.startswith('Title:'):
            current_topic['title'] = line.replace('Title:', '').strip()
        elif line.startswith('Description:'):
            current_topic['description'] = line.replace('Description:', '').strip()
        elif line.startswith('Narrative:'):
            current_topic['narrative'] = line.replace('Narrative:', '').strip()

    if current_topic:
        topics.append(current_topic)

    return topics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert topics to NIST format')
    parser.add_argument('--input', required=True,
                        help='Input topics file')
    parser.add_argument('--output', required=True,
                        help='Output XML file')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'],
                        default='json',
                        help='Input format')

    args = parser.parse_args()

    convert_to_nist_format(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
