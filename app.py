from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	## Read Data
	df=pd.read_csv('dataProblem.csv')
	# Features and Labels
	df['label'] = df['class'].map({'stmt': 0, 'problem': 1})
	X = df['sent']
	y = df['label']
	
	"""##Data Split"""
	Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X,y,test_size=0.3, random_state=42, shuffle=True)
	
	"""##Vecorization"""
	#make dataset's words as features with the vectorizer
	#vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,2), min_df=1, max_df=1.0) # 77.71
	vectorizer = CountVectorizer() # 78.37
	vectorizer.fit(X)
	#Vectorization
	Train_X_vectorized = vectorizer.transform(Train_X)
	Test_X_vectorized = vectorizer.transform(Test_X)
	#instanciate Logistic Regression Model
	classifier = LogisticRegression(solver='lbfgs', max_iter=2000)
	#Train model
	classifier.fit(Train_X_vectorized,Train_Y)
	prediction_logreg = classifier.predict(Test_X_vectorized)
	#evaluate model
	M=round(accuracy_score(prediction_logreg, Test_Y)*100, 2)
	print("Logistic Regression Accuracy Score: {}%".format(M))
	

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = vectorizer.transform(data).toarray()
		my_prediction = classifier.predict(vect)
	return render_template('home.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)