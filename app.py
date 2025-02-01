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

def apply_final_transformation(dataset_file, harmonization_report_file):
    print("✅ Applying final transformation...")

    harmonization_df = pd.read_excel(harmonization_report_file)
    harmonization_df = harmonization_df.dropna(subset=['Matched Term'])  # Keep only matched features
    harmonization_df = harmonization_df[harmonization_df['Matched Term'] != "No match found"]

    input_filename = os.path.splitext(dataset_file.filename)[0]
    dataset_ext = os.path.splitext(dataset_file.filename)[-1].lower()
    
    if dataset_ext == ".csv":
        dataset = pd.read_csv(dataset_file, encoding="utf-8")
        print("✅ Successfully loaded dataset as CSV file.")
    elif dataset_ext in [".xls", ".xlsx"]:
        dataset = pd.read_excel(dataset_file)
        print("✅ Successfully loaded dataset as Excel file.")
    else:
        print("❌ Unsupported dataset format:", dataset_ext)
        return None, "Unsupported dataset format!"

    transformed_dataset = dataset.copy()
    columns_to_keep = []

    for _, row in harmonization_df.iterrows():
        original_name = row["Feature Name"]
        matched_name = row["Matched Term"]
        target_range = row.get("Target Value Range", "")

        if original_name not in transformed_dataset.columns:
            print(f"⚠️ Column {original_name} not found in dataset. Skipping.")
            continue

        transformed_dataset.rename(columns={original_name: matched_name}, inplace=True)
        columns_to_keep.append(matched_name)

        try:
            # Convert value range strings to lists
            value_range = eval(row["Value Range"]) if isinstance(row["Value Range"], str) else row["Value Range"]
            target_value_range = eval(target_range) if isinstance(target_range, str) else target_range

            if isinstance(value_range, list) and isinstance(target_value_range, list) and len(value_range) == len(target_value_range):
                mapping_dict = {str(k): v for k, v in zip(value_range, target_value_range)}
                
                print(f"🔎 Mapping for {matched_name}: {mapping_dict}")

                # Apply mapping only to non-missing values
                mask = transformed_dataset[matched_name].notna()
                transformed_dataset.loc[mask, matched_name] = transformed_dataset.loc[mask, matched_name].astype(str).map(mapping_dict)

                print(f"Before mapping: {dataset[original_name].value_counts(dropna=False)}")
                print(f"After mapping: {transformed_dataset[matched_name].value_counts(dropna=False)}")
                print(f"✅ Transformed {matched_name} successfully.")

            else:
                print(f"⚠️ Skipping transformation for {matched_name} (Invalid Mapping)")
        except Exception as e:
            print(f"⚠️ Error transforming {matched_name}: {e}")
            continue

    transformed_dataset = transformed_dataset[columns_to_keep]
    
    transformed_filename = f"{input_filename}_harmonized_dataset.xlsx"
    transformed_path = os.path.join(app.config['RESULTS_FOLDER'], transformed_filename)
    transformed_dataset.to_excel(transformed_path, index=False)
    print(f"✅ Final harmonized dataset saved: {transformed_path}")
    return transformed_filename, "Final harmonization completed successfully!"

@app.route('/main', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("🔹 Received POST request!")

        action = request.form.get('action')
        print("User selected:", action)

        if not action:
            print("❌ No action received!")
            return render_template('index.html', success=False, message="No action selected!")

        if action == "metadata_harmonization":
            report_file = request.files.get('report')
            xml_file = request.files.get('xml')

            if not report_file or not xml_file:
                print("❌ Missing required files for metadata harmonization!")
                return render_template('index.html', success=False, message="Missing required files!")

            print("✅ Processing metadata harmonization...")

            report_path = os.path.join(app.config['RESULTS_FOLDER'], report_file.filename)
            xml_path = os.path.join(app.config['RESULTS_FOLDER'], xml_file.filename)
            report_file.save(report_path)
            xml_file.save(xml_path)

            metadata = extract_metadata_from_report(report_path)
            semantic_knowledge, child_to_parent_map = extract_semantic_knowledge_from_xml(xml_path)
            corpus_mapping = load_corpus()

            terminology_dict = {term['tag']: {'synonyms': [], 'subclasses': []} for term in semantic_knowledge}

            input_filename = os.path.splitext(os.path.basename(report_file.filename))[0]
            matching_results = perform_matching(metadata, terminology_dict, child_to_parent_map, corpus_mapping)
            harmonization_report_path = generate_harmonization_report(matching_results, input_filename)

            print("✅ Metadata harmonization completed!")
            return render_template('index.html', success=True, message='Metadata harmonization report generated!', harmonized_file=harmonization_report_path)


        elif action == "final_harmonization":
            harmonization_report_file = request.files.get('harmonization_report')
            dataset_file = request.files.get('dataset')

            if not harmonization_report_file or not dataset_file:
                print("❌ Missing required files for final harmonization!")
                return render_template('index.html', success=False, message="Missing required files!")

            print("✅ Processing final harmonization...")
            transformed_file, message = apply_final_transformation(dataset_file, harmonization_report_file)
            return render_template('index.html', success=True, message=message)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
