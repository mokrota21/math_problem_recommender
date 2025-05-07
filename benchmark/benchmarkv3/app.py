from flask import Flask, request, redirect, url_for, render_template, flash
import pandas as pd
import os
from labeling import LabelingTool

# Set up paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'math_problem_recommender_secret_key'  # Required for flash messages
tool = LabelingTool()

@app.route('/')
def index():
    subtopics = tool.get_subtopics()
    return render_template('subtopics.html', subtopics=subtopics)

@app.route('/submit_labels', methods=['POST'])
def submit_labels():
    # Will collect the selected/sample and new rows
    ids = {}
    new_rows_indices = {}
    
    # For each label, determine if sampled or new row
    for label in ['Anchor', 'Golden', 'Silver', 'Wrong']:
        section_choice = request.form.get(f"section_choice_{label}", 'sample')
        
        if section_choice == 'sample':
            # Use the radio button value for this label
            if label in request.form:
                ids[label] = int(request.form.get(label))
        elif section_choice == 'new':
            # Gather new row data from the textboxes
            new_row = {}
            # Get columns from the tool's df
            cols = list(tool.label_dfs[label].columns)
            for col in cols:
                field_name = f"new_{label}_{col}"
                new_row[col] = request.form.get(field_name, '')
            
            # Add the new row and get its index
            new_idx = tool.add_row(label, new_row)
            ids[label] = new_idx
            if label not in new_rows_indices:
                new_rows_indices[label] = []
            new_rows_indices[label].append(new_idx)
    
    query = request.form.get('query', '')
    print('Selected IDs:', ids)
    print('New row indices:', new_rows_indices)
    
    # Make sure we have all required labels
    if len(ids) == 4:  # All four labels are present
        tool.add_qa(ids, query)
        tool.save()
        return redirect(url_for('index'))
    else:
        # Handle the case where not all labels were selected
        missing = set(['Anchor', 'Golden', 'Silver', 'Wrong']) - set(ids.keys())
        flash(f"Please select or create entries for all sections. Missing: {', '.join(missing)}")
        return redirect(url_for('label_subtopic', subtopic=request.form.get('subtopic')))



@app.route('/resample_table/<label>', methods=['POST'])
def resample_table(label):
    subtopic = (request.json or {}).get('subtopic')
    tool.resample(label)
    df = tool.label_dfs[label]
    return render_template('table_snippet.html', df=df, label=label)

@app.route('/label', methods=['GET'])
def label_subtopic():
    subtopic = request.args.get('subtopic')
    tool.labeling_options_init(subtopic)
    tables = tool.label_dfs
    return render_template('labeling.html', subtopic=subtopic, tables=tables)

@app.route('/resample/<label>', methods=['POST'])
def resample(label):
    subtopic = request.form.get('subtopic')
    tool.resample(label)
    tables = tool.label_dfs
    return render_template('labeling.html', subtopic=subtopic, tables=tables)

if __name__ == '__main__':
    app.run(debug=True)
