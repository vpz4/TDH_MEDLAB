from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
import sys
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = "results"
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
CORPUS_PATH = "corpus_REDUCED.csv"

def extract_metadata_from_report(report_path):
    print("Extracting metadata from report...")
    report_df = pd.read_excel(report_path, skiprows=12)
    report_df = report_df.iloc[:, :2]
    report_df.columns = ['Feature Name', 'Value Range']
    return report_df.to_dict('records')

def extract_semantic_knowledge_from_xml(xml_path):
    print("Extracting semantic knowledge from XML...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [{'tag': elem.tag.lower(), 'attributes': elem.attrib, 'source': 'XML'} for elem in root.iter() if elem.tag != root.tag]

def load_and_enrich_corpus():
    print("Loading and enriching hardcoded vocabulary corpus...")
    corpus_df = pd.read_csv(CORPUS_PATH, dtype=str, low_memory=False)
    enriched_corpus = []
    for _, row in corpus_df.iterrows():
        term = str(row.get("concept_name", ""))
        concept_id = str(row.get("concept_id", "Unknown"))
        source = str(row.get("vocabulary_id", "Corpus"))
        synonyms = set(lemma.name() for syn in wordnet.synsets(term) for lemma in syn.lemmas())
        enriched_corpus.append({'term': term, 'synonyms': list(synonyms), 'id': concept_id, 'source': source})
    return enriched_corpus

def perform_matching(metadata, enriched_corpus, semantic_knowledge):
    print("Performing lexical and semantic matching...")
    vectorizer = TfidfVectorizer()
    matching_results = []
    xml_terms = {item['tag']: item for item in semantic_knowledge}

    for feature in metadata:
        feature_name = feature['Feature Name'].lower()
        best_match = None
        best_score = 0

        for xml_term, xml_data in xml_terms.items():
            score = max(fuzz.ratio(feature_name, xml_term), fuzz.WRatio(feature_name, xml_term)) / 100
            if score > best_score:
                best_match = {'Feature Name': feature['Feature Name'], 'Matched Term': xml_term, 'Matching Score': score, 'Source': 'XML'}
                best_score = score

        tfidf_matrix = vectorizer.fit_transform([feature_name] + [item['term'] for item in enriched_corpus])
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        for i, score in enumerate(similarities):
            if score > best_score:
                best_match = {'Feature Name': feature['Feature Name'], 'Matched Term': enriched_corpus[i]['term'], 'Matching Score': score, 'Source': enriched_corpus[i]['source'], 'Reference ID': enriched_corpus[i]['id']}
                best_score = score

        if best_match:
            matching_results.append(best_match)
    
    return matching_results

def generate_harmonization_report(matching_results, input_filename):
    print("Generating harmonization report...")
    report_filename = f"{input_filename}_data_harmonization_report.xlsx"
    file_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
    pd.DataFrame(matching_results).to_excel(file_path, index=False)
    return file_path

def apply_final_harmonization(input_dataset_path, harmonization_report_path):
    print("Applying final harmonization to dataset...")
    dataset = pd.read_csv(input_dataset_path)
    harmonization_df = pd.read_excel(harmonization_report_path)
    rename_mapping = {row['Feature Name']: row['Matched Term'] for _, row in harmonization_df.iterrows() if row['Feature Name'] in dataset.columns}
    dataset.rename(columns=rename_mapping, inplace=True)
    output_harmonized_path = os.path.splitext(input_dataset_path)[0] + "_harmonized.csv"
    dataset.to_csv(output_harmonized_path, index=False)
    return output_harmonized_path

@app.route('/main', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        print("User selected:", action)
        
        if action == "metadata_harmonization":
            report_file = request.files['report']
            xml_file = request.files['xml']
            metadata = extract_metadata_from_report(report_file)
            semantic_knowledge = extract_semantic_knowledge_from_xml(xml_file)
            enriched_corpus = load_and_enrich_corpus()
            input_filename = os.path.splitext(os.path.basename(report_file.filename))[0]
            matching_results = perform_matching(metadata, enriched_corpus, semantic_knowledge)
            harmonization_report_path = generate_harmonization_report(matching_results, input_filename)
            sys.stdout.flush()
            return render_template('index.html', success=True, message='Metadata harmonization report generated!', harmonized_file=harmonization_report_path)
        
        elif action == "final_harmonization":
            harmonization_report_file = request.files['harmonization_report']
            dataset_file = request.files['dataset']
            harmonized_dataset_path = apply_final_harmonization(dataset_file, harmonization_report_file)
            return render_template('index.html', success=True, message='Final harmonization applied!', harmonized_file=harmonized_dataset_path)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)