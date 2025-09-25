import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
from urllib.parse import urljoin, urlparse

def clean_text(text):
    """
    Replaces <code> tags with spaces in the text.

    Args:
        text (str): Text to be cleaned

    Returns:
        str: Cleaned text
    """
    cleaned_text = re.sub(r'<code>', r' ', text)
    cleaned_text = re.sub(r'</code>', r' ', cleaned_text)
    cleaned_text = re.sub(r'<[^>]*>', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def is_subpath(base_url, target_url):
    """Check if target_url is a subpath of base_url"""
    base = urlparse(base_url)
    target = urlparse(urljoin(base_url, target_url))
    
    if base.netloc == target.netloc:
        is_subpath_check = True
    else: 
        is_subpath_check = False

    return is_subpath_check

def extract_text_from_link(base_url, link_url):
    """
    Extracts text content from a hyperlink.

    Args:
        base_url (str): Base URL
        link_url (str): Relative or absolute link URL

    Returns:
        str: Extracted text content
    """
    full_url = urljoin(base_url, link_url)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(full_url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"[Failed to access link: Status code {response.status_code}]"

        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('article') or soup.body

        if not main_content:
            return "[Main content not found]"

        for script in main_content(["script", "style"]):
            script.decompose()
        
        bread_info_all = main_content.find_all('li', class_='breadcrumb-item')
        if bread_info_all:
            for bread_info in bread_info_all:
                bread_info.decompose()

        bread_active_info = main_content.find('li', class_='breadcrumb-item active')
        if bread_active_info:
            bread_active_info.decompose()

        feedback_div = main_content.find('div', class_='d-print-none')
        if feedback_div:
            feedback_div.decompose()
        
        cite_save_info = main_content.find('div', class_='text-muted mt-5 pt-3 border-top')
        if cite_save_info:
            cite_save_info.decompose()
        
        # table format
        table_all = main_content.find_all('table')
        table_text = ""
        if table_all:
            for table in table_all:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows = [[td.get_text(strip=True) for td in row.find_all('td')]
                        for row in table.find_all('tr') if row.find_all('td')]

                # make text table
                col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
                def format_row(row): return "| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
                divider = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

                table_lines = ["Table columns:", divider, format_row(headers), divider]
                table_lines += [format_row(row) for row in rows]
                table_lines.append(divider)
                table_text += "\n" + "\n".join(table_lines) + "\n"
                table.decompose()
        text = main_content.get_text(separator='').strip()
        text = re.sub(r'\n\n', ' ', text)

        return f"{table_text}\n\n{text}"

    except Exception as e:
        return f"[Error accessing link: {str(e)}]"

def scrape_mimic_tables_info(url):
    """
    Scrapes text and hyperlinks from the MIMIC-III Tables web page.

    Args:
        url (str): Webpage URL to scrape

    Returns:
        DataFrame: DataFrame containing text and link information
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to retrieve webpage. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    table_info = []
    main_content = soup.find('main') or soup.find('div', class_='content') or soup

    headers = main_content.find_all(['h1', 'h2', 'h3', 'h4'])
    for header in headers:
        section_title = header.get_text(strip=True)

        if "Feedback" in section_title:
            continue

        next_element = header.find_next_sibling()

        while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4']:
            if next_element.name in ['p', 'table']:
                html_content = str(next_element)
                text_content = clean_text(html_content)
                links = next_element.find_all('a')
                link_info = []

                for link in links:
                    link_text = link.get_text(strip=True)
                    link_url = link.get('href', '')

                    if is_subpath(url, link_url):
                        link_content = extract_text_from_link(url, link_url)
                        link_info.append((link_text, link_url, link_content))

                if text_content:
                    table_info.append({
                        'section': header.get_text(strip=True),
                        'text': text_content,
                        'links': link_info
                    })

            elif next_element.name in ['ul', 'ol']:
                list_items = next_element.find_all('li')
                for li in list_items:
                    html_content = str(li)
                    li_text = clean_text(html_content)
                    links = li.find_all('a')
                    link_info = []

                    for link in links:
                        link_text = link.get_text(strip=True)
                        link_url = link.get('href', '')

                        if is_subpath(url, link_url):
                            link_content = extract_text_from_link(url, link_url)
                            link_info.append((link_text, link_url, link_content))

                    if li_text:
                        table_info.append({
                            'section': header.get_text(strip=True),
                            'text': f"- {li_text}",
                            'links': link_info
                        })

            elif next_element.name == 'div' and 'entry' in next_element.get('class', []):
                h5_tag = next_element.find('h5')
                if h5_tag:
                    a_tag = h5_tag.find('a')
                    if a_tag:
                        link_text = a_tag.get_text(strip=True)
                        link_url = a_tag.get('href', '')

                        if is_subpath(url, link_url):
                            link_content = extract_text_from_link(url, link_url)
                            table_info.append({
                                'section': section_title,
                                'text': f"{link_text}",
                                'links': [(link_text, link_url, link_content)]
                            })

            next_element = next_element.find_next_sibling()

    entry_divs = main_content.find_all('div', class_='entry')

    for entry_div in entry_divs:
        h5_tag = entry_div.find('h5')
        if h5_tag:
            a_tag = h5_tag.find('a')
            if a_tag:
                link_text = a_tag.get_text(strip=True)
                link_url = a_tag.get('href', '')

                if is_subpath(url, link_url):
                    section_name = "Table Information"
                    link_content = extract_text_from_link(url, link_url)
                    table_info.append({
                        'section': section_name,
                        'text': f"{link_text}",
                        'links': [(link_text, link_url, link_content)]
                    })

    df = pd.DataFrame(table_info)
    return df

def save_to_txt(df, filename="mimic_tables_info.txt"):
    """Save results to a TXT file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== MIMIC-III Tables Information ===\n\n")

        current_section = []
        for idx, row in df.iterrows():
            if row['section'] not in current_section:
                current_section.append(row['section'])
                f.write(f"\n\n[{row['section']}]\n")

            f.write(f"{row['text']}\n")

            if row['links']:
                for link_text, link_url, link_content in row['links']:
                    f.write(f" {link_content}\n\n")
    print(f"Results saved to {filename}.")

if __name__ == "__main__":
    code_path = "/home/jovyan/nfs-thena/winter_intern/Few-shot-NL2SQL-with-prompting/data"
    os.chdir(code_path)

    url = "https://mimic.mit.edu/docs/iii/tables/"
    try:
        print(f"Extracting information from {url}...")
        result_df = scrape_mimic_tables_info(url)

        print("\n=== Extracted Information ===")
        section_list = []
        for idx, row in result_df.iterrows():
            if row['section'] not in section_list:
                section_list.append(row['section'])
                print(f"\n[{row['section']}]")
            print(f"{row['text']}")

            if row['links']:
                for link_text, link_url, link_content in row['links']:
                    print(f" {link_content[:100]}...")

        save_to_txt(result_df)
        print("\nScraping completed!")

    except Exception as e:
        print(f"Error occurred: {e}")
