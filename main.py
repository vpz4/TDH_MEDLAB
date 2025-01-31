# -*- coding: utf-8 -*-
"""
Created on Wed Sep  12 16:55:01 2023

@author: vpezoulas
"""

import nltk
import os
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Extract metadata from data quality report
def extract_metadata_from_report(report_path):
    report_df = pd.read_excel(report_path, skiprows=12)
    report_df = report_df.iloc[:, :2]
    report_df.columns = ['Feature Name', 'Value Range']

    metadata = report_df.to_dict('records')
    return metadata

# 2. Extract semantic knowledge from XML data model
def extract_semantic_knowledge_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    semantic_knowledge = []
    for elem in root.iter():
        if elem.tag != root.tag:
            attributes = {
                'tag': elem.tag,
                'attributes': elem.attrib,
                'source': 'XML'
            }
            semantic_knowledge.append(attributes)
    return semantic_knowledge

# 3. Load and enrich vocabulary corpus
def load_and_enrich_corpus(corpus_path):
    corpus_df = pd.read_csv(corpus_path, dtype=str, low_memory=False)  # Read CSV with consistent data types

    print("Available columns:", corpus_df.columns)

    actual_term_column = "concept_name"
    id_column = "concept_id"
    source_column = "vocabulary_id"
    
    enriched_corpus = []
    for _, row in corpus_df.iterrows():
        term = str(row[actual_term_column]) if pd.notna(row[actual_term_column]) else ""
        concept_id = str(row[id_column]) if id_column in corpus_df.columns else "Unknown"
        source = str(row[source_column]) if source_column in corpus_df.columns else "Corpus"
        
        if not term.strip():
            continue
        
        synonyms = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        enriched_corpus.append({'term': term, 'synonyms': list(synonyms), 'id': concept_id, 'source': source})
    
    return enriched_corpus

# 4. Lexical/semantic matching
def perform_matching(metadata, enriched_corpus, semantic_knowledge):
    vectorizer = TfidfVectorizer()
    matching_results = []
    xml_terms = {item['tag'].lower(): item for item in semantic_knowledge}

    for feature in metadata:
        feature_name = feature['Feature Name']
        print("Working for feature", feature_name)
        
        # Check for high Levenshtein or Jaro similarity with XML terms
        for xml_term, xml_data in xml_terms.items():
            levenshtein_score = fuzz.ratio(feature_name.lower(), xml_term)
            jaro_score = fuzz.WRatio(feature_name.lower(), xml_term)
            
            if max(levenshtein_score, jaro_score) > 85:  # Threshold for similarity
                matching_results.append({
                    'Feature Name': feature_name,
                    'Matched Term': xml_data['tag'],
                    'Matching Score': max(levenshtein_score, jaro_score) / 100,
                    'Source': 'XML',
                    'Reference ID': xml_data.get('attributes', {}).get('concept_id', 'Unknown')
                })
                continue

        # TF-IDF Matching
        tfidf_matrix = vectorizer.fit_transform(
            [feature_name] + [item['term'] for item in enriched_corpus]
        )

        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        for i, score in enumerate(similarities):
            if score > 0.7:  # Threshold for relevance
                print({
                    'Feature Name': feature_name,
                    'Matched Term': enriched_corpus[i]['term'],
                    'Matching Score': score,
                    'Source': enriched_corpus[i]['source'],
                    'Reference ID': enriched_corpus[i]['id']  # Now correctly mapping to concept_id
                })

                matching_results.append({
                    'Feature Name': feature_name,
                    'Matched Term': enriched_corpus[i]['term'],
                    'Matching Score': score,
                    'Source': enriched_corpus[i]['source'],
                    'Reference ID': enriched_corpus[i]['id']  # Now correctly mapping to concept_id
                })

    return matching_results

# 5. Generate harmonization report
def generate_harmonization_report(matching_results, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, "data_harmonization_report.xlsx")
    report_df = pd.DataFrame(matching_results)
    report_df.to_excel(file_path, index=False)
    print(f"Harmonization report saved to {file_path}")

# Main script execution
def main():
    report_path = input("Enter the path to the data quality evaluation report (.xlsx): ").strip()
    while not os.path.isfile(report_path) or not report_path.endswith('.xlsx'):
        print("Invalid file. Please provide a valid .xlsx file.")
        report_path = input("Enter the path to the data quality evaluation report (.xlsx): ").strip()

    xml_path = input("Enter the path to the data model (.xml or .owl): ").strip()
    while not os.path.isfile(xml_path) or not (xml_path.endswith('.xml') or xml_path.endswith('.owl')):
        print("Invalid file. Please provide a valid .xml or .owl file.")
        xml_path = input("Enter the path to the data model (.xml or .owl): ").strip()

    corpus_path = "test/corpus_REDUCED.csv"

    # Set output folder
    output_folder = os.path.join(os.getcwd(), "results")

    # Step 1: Extract metadata
    metadata = extract_metadata_from_report(report_path)

    # Step 2: Extract semantic knowledge
    semantic_knowledge = extract_semantic_knowledge_from_xml(xml_path)

    # Step 3: Load and enrich corpus
    enriched_corpus = load_and_enrich_corpus(corpus_path)

    # Step 4: Perform matching
    matching_results = perform_matching(metadata, enriched_corpus, semantic_knowledge)

    # Step 5: Generate harmonization report
    generate_harmonization_report(matching_results, output_folder)

if __name__ == "__main__":
    main()