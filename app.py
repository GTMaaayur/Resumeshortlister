from flask import Flask, request, render_template, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ProcessPoolExecutor
import logging

app = Flask(__name__)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error Extracting text:{str(e)}")
        return None

def process_batch(job_description_file, resume_files):
    try:
        job_description_text = extract_text_from_pdf(job_description_file)
        if not job_description_text:
            return None, {'error': 'Failed to extract text from job description PDF'}

        job_embedding = model.encode(job_description_text, convert_to_tensor=True)

        name_scores = []
        errors = []

        for resume_file in resume_files:
            try:
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text:
                    errors.append({'resume': resume_file.filename, 'error': 'Failed to extract text from resume PDF'})
                    continue

                resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                similarity_score = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0]
                name_scores.append({'resume': resume_file.filename, 'score': similarity_score.item()})
            except Exception as e:
                errors.append({'resume': resume_file.filename, 'error': str(e)})

        return name_scores, errors

    except Exception as e:
        return None, {'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_resumes', methods=['POST'])
def analyze_resumes():
    if 'job_description' not in request.files or 'resumes[]' not in request.files:
        return jsonify({'error': 'Missing files'})

    job_description_file = request.files['job_description']
    resume_files = request.files.getlist('resumes[]')

    allowed_formats = ['.pdf']
    if not job_description_file.filename.endswith(tuple(allowed_formats)) or any(
            not resume.filename.endswith(tuple(allowed_formats)) for resume in resume_files):
        return jsonify({'error': 'Invalid file format. Only PDF files are allowed.'})

    errors = []
    name_scores = []

    for resume_file in resume_files:
        try:
            job_description_text = extract_text_from_pdf(job_description_file)
            resume_text = extract_text_from_pdf(resume_file)

            if not job_description_text or not resume_text:
                errors.append({'resume': resume_file.filename, 'error': 'Failed to extract text'})
                continue

            job_embedding = model.encode(job_description_text, convert_to_tensor=True)
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)

            similarity_score = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0]
            name_scores.append({'resume': resume_file.filename, 'score': similarity_score.item()})
        except Exception as e:
            errors.append({'resume': resume_file.filename, 'error': str(e)})

    if errors:
        return jsonify({'error': 'Errors encountered', 'errors': errors})
    else:
        sorted_name_scores = sorted(name_scores, key=lambda item: item['score'], reverse=True)
        resume_output = [entry['resume'] for entry in sorted_name_scores]
        scores_output = [entry['score'] for entry in sorted_name_scores]

        return jsonify({'resume': resume_output, 'scores': scores_output})


if __name__ == "__main__":
    app.run(debug=True)
