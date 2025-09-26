import requests
import pandas as pd
import csv
import re
import os
from bs4 import BeautifulSoup

def extract_text_from_website(url):
    print(f"Extracting information from {url}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        page_title = soup.title.string if soup.title else "제목 없음"
        print(f"페이지 제목: {page_title}")
        
        main_content = soup.find('main') if soup.find('main') else soup
        
        for element in main_content.find_all(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        results = []
        
        headers = main_content.find_all(['h1', 'h2', 'h3', 'h4'])
        current_section = "General"
        
        first_header = headers[0] if headers else None
        if first_header:
            prev_elements = []
            for sibling in first_header.previous_siblings:
                if sibling.name in ['p', 'ul', 'ol', 'div', 'table']:
                    prev_elements.append(sibling)
            
            for element in reversed(prev_elements):
                if element.name == 'table':
                    table_text = process_table(element)
                    if table_text:
                        results.append({"section": current_section, "text": table_text})
                elif element.name in ['p', 'ul', 'ol', 'div']:
                    text = element.get_text(strip=True)
                    if text:
                        results.append({"section": current_section, "text": text})
        
        for i, header in enumerate(headers):
            current_section = header.get_text(strip=True)
            results.append({"section": current_section, "text": current_section})
            
            next_header = headers[i+1] if i+1 < len(headers) else None
            
            elements = []
            for sibling in header.next_siblings:
                if next_header and sibling == next_header:
                    break
                if sibling.name in ['p', 'ul', 'ol', 'li', 'div', 'table']:
                    elements.append(sibling)
            
            for element in elements:
                if element.name == 'table':
                    table_text = process_table(element)
                    if table_text:
                        results.append({"section": current_section, "text": table_text})
                elif element.name in ['p', 'ul', 'ol', 'li', 'div']:
                    text = element.get_text(strip=True)
                    if text:
                        results.append({"section": current_section, "text": text})
        
        if headers:
            last_header = headers[-1]
            elements = []
            for sibling in last_header.next_siblings:
                if sibling.name in ['p', 'ul', 'ol', 'li', 'div', 'table']:
                    elements.append(sibling)
            
            for element in elements:
                if element.name == 'table':
                    table_text = process_table(element)
                    if table_text:
                        results.append({"section": current_section, "text": table_text})
                elif element.name in ['p', 'ul', 'ol', 'li', 'div']:
                    text = element.get_text(strip=True)
                    if text:
                        results.append({"section": current_section, "text": text})
        
        result_df = pd.DataFrame(results)
        
        return result_df
        
    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")
        return pd.DataFrame()

def process_table(table):
    """테이블을 텍스트로 처리하는 함수"""
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = [[td.get_text(strip=True) for td in row.find_all('td')]
            for row in table.find_all('tr') if row.find_all('td')]
    
    # 테이블이 비어있으면 처리하지 않음
    if not headers or not rows:
        return None
    
    # 텍스트 테이블 만들기
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
    def format_row(row): return "| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
    divider = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    
    table_lines = [divider, format_row(headers), divider]
    table_lines += [format_row(row) for row in rows]
    table_lines.append(divider)
    
    return "\n".join(table_lines)

def save_to_txt(dataframe):
    """결과를 TXT 파일로 저장"""
    filename = "eicu_tables_info.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        section_list = []
        for idx, row in dataframe.iterrows():
            if row['section'] not in section_list:
                section_list.append(row['section'])
                file.write(f"\n[{row['section']}]\n")
            file.write(f"{row['text']}\n")
    print(f"TXT 파일이 저장되었습니다: {filename}")

if __name__ == "__main__":

    code_path = "/home/jovyan/nfs-thena/winter_intern/Few-shot-NL2SQL-with-prompting/data"
    os.chdir(code_path)

    overview_url = "https://eicu-crd.mit.edu/gettingstarted/overview/"
    all_table = [
    "admissiondrug",
    "admissiondx",
    "allergy",
    "apacheapsvar",
    "apachepatientresult",
    "apachepredvar",
    "apacheprepatientresult",
    "careplancareprovider",
    "careplaneol",
    "careplangeneral",
    "careplangoal",
    "careplaninfectiousdisease",
    "customlab",
    "diagnosis",
    "infusiondrug",
    "intakeoutput",
    "lab",
    "medication",
    "microlab",
    "note",
    "nurseassessment",
    "nursecare",
    "nursecharting",
    "pasthistory",
    "patient",
    "physicalexam",
    "respiratorycare",
    "respiratorycharting",
    "treatment",
    "vitalaperiodic",
    "vitalperiodic"
    ]

    result_df = extract_text_from_website(overview_url)
    for table_url in all_table:
        table_df = extract_text_from_website(f"https://eicu-crd.mit.edu/eicutables/{table_url}/")
        result_df = pd.concat([result_df,table_df])
    
    section_list = []
    for idx, row in result_df.iterrows():
        if row['section'] not in section_list:
            section_list.append(row['section'])
            print(f"\n[{row['section']}]")
        print(f"{row['text']}")
    
    save_to_txt(result_df)
    print("\nScraping completed!")