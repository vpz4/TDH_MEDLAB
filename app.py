from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
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
    child_to_parent_map = {}

    for elem in root.iter():
        for child in elem:
            child_to_parent_map[child.tag.lower()] = elem.tag.lower()

    return [{'tag': elem.tag.lower(), 'attributes': elem.attrib} for elem in root.iter()], child_to_parent_map

def load_corpus():
    print("Loading vocabulary corpus...")
    corpus_df = pd.read_csv(CORPUS_PATH, dtype=str, low_memory=False)
    corpus_mapping = {row["concept_name"].lower(): {"id": row["concept_id"], "name": row["concept_name"]}
                      for _, row in corpus_df.iterrows() if pd.notna(row["concept_name"])}
    return corpus_mapping

def perform_matching(metadata, terminology_dict, child_to_parent_map, corpus_mapping):
    print("Performing lexical and semantic matching...")
    vectorizer = TfidfVectorizer()
    matching_results = []
    similarity_threshold = 0.5  

    for feature in metadata:
        feature_name = feature['Feature Name'].lower()
        value_range = feature['Value Range']
        best_match = None
        best_score = similarity_threshold  
        parent_name = "N/A"

        print(f"\nProcessing feature: {feature_name}")

        # Step 1: Check for Exact Match in XML Terminology
        for term, data in terminology_dict.items():
            term_variants = [term] + data['synonyms'] + data['subclasses']
            if feature_name in term_variants:
                print(f"✅ Exact match found: {feature_name} -> {term} (Source: XML)")
                parent_name = child_to_parent_map.get(term, "No Parent")
                best_match = {
                    'Feature Name': feature['Feature Name'],
                    'Matched Term': term,
                    'Matching Score': 1.0,
                    'Value Range': value_range,
                    'Source': 'XML',
                    'Parent Term': parent_name
                }
                matching_results.append(best_match)
                print(f"✅ Stored exact match: {best_match}")
                break

        if best_match:
            continue

        # Step 2: Apply similarity-based matching
        corpus_terms = list(corpus_mapping.keys())
        tfidf_matrix = vectorizer.fit_transform([feature_name] + corpus_terms)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        for i, score in enumerate(similarities):
            if score > best_score:
                matched_corpus_term = corpus_terms[i]
                mapped_xml_info = corpus_mapping.get(matched_corpus_term, {"id": "Unknown", "name": "Unknown"})
                matched_xml_term = mapped_xml_info['name'] if mapped_xml_info['name'] != "Unknown" else matched_corpus_term
                best_match = {
                    'Feature Name': feature['Feature Name'],
                    'Matched Term': matched_xml_term,
                    'Matching Score': score,
                    'Value Range': value_range,
                    'Source': 'Corpus -> XML',
                    'Concept ID': mapped_xml_info['id']
                }
                parent_name = child_to_parent_map.get(matched_xml_term, "No Parent")
                best_match['Parent Term'] = parent_name
                best_score = score

        if best_match:
            matching_results.append(best_match)
            print(f"✅ Stored match: {best_match}")
        else:
            no_match_entry = {
                'Feature Name': feature['Feature Name'],
                'Matched Term': "No match found",
                'Matching Score': 0,
                'Value Range': value_range,
                'Source': 'N/A',
                'Concept ID': 'N/A',
                'Parent Term': "N/A"
            }
            matching_results.append(no_match_entry)
            print(f"❌ No match found, storing: {no_match_entry}")

    return matching_results

def generate_harmonization_report(matching_results, input_filename):
    print("Generating harmonization report...")
    report_filename = f"{input_filename}_data_harmonization_report.xlsx"
    file_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
    pd.DataFrame(matching_results).to_excel(file_path, index=False)
    return file_path

@app.route('/main', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("🔹 Received POST request!")

        action = request.form.get('action')
        print("User selected:", action)

        if not action:
            print("❌ No action received!")
            return render_template('index.html', success=False, message="No action selected!")

        report_file = request.files.get('report')
        xml_file = request.files.get('xml')

        if action == "metadata_harmonization":
            if not report_file or not xml_file:
                print("❌ Missing required files for metadata harmonization!")
                return render_template('index.html', success=False, message="Missing required files!")

            print("✅ Processing metadata harmonization...")
            metadata = extract_metadata_from_report(report_file)
            semantic_knowledge, child_to_parent_map = extract_semantic_knowledge_from_xml(xml_file)
            corpus_mapping = load_corpus()

            terminology_dict = {term['tag']: {'synonyms': [], 'subclasses': []} for term in semantic_knowledge}

            input_filename = os.path.splitext(os.path.basename(report_file.filename))[0]
            matching_results = perform_matching(metadata, terminology_dict, child_to_parent_map, corpus_mapping)
            harmonization_report_path = generate_harmonization_report(matching_results, input_filename)

            print("✅ Metadata harmonization completed!")
            return render_template('index.html', success=True, message='Metadata harmonization report generated!', harmonized_file=harmonization_report_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
