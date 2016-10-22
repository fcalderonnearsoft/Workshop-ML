from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import pickle

app = Flask(__name__)

######## Preparing the Predictor
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects/diabetes.pkl'), 'rb'))

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    print ("==========================================")
    print (document)
    print ("==========================================")
    document = document.split(',')
    document = [float(i) for i in document]
    y = clf.predict([document])[0]
    return label[y]


class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST':
        pregnacies = request.form['number_of_pregnacies']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        thickness = request.form['thickness']
        insulin = request.form['insulin']
        body_mass_index = request.form['body_mass_index']
        diabetes_pedigree = request.form['diabetes_pedigree']
        age = request.form['age']
        test = pregnacies+ "," + glucose+ "," + blood_pressure+ "," + thickness+ "," + insulin+ "," + body_mass_index+ "," + diabetes_pedigree+ "," + age
        y = classify(test)
        return render_template('results.html',
                                content=test,
                                prediction=y)
    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)

    #
    #
    #2,108,64,30.37974684,156.05084746,30.8,0.158,21
