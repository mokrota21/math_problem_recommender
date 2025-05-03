from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
import os
from labeling import LabelingTool

# Set up paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
tool = LabelingTool()

@app.route('/')
def index():
    subtopics = tool.get_subtopics()
    return render_template('subtopics.html', subtopics=subtopics)

@app.route('/submit_labels', methods=['POST'])
def submit_labels():
    ids = {label: int(request.form[label]) for label in ['Anchor', 'Golden', 'Silver', 'Wrong']}
    print(ids)
    tool.add_qa(ids)
    tool.save()
    return redirect(url_for('index'))

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
