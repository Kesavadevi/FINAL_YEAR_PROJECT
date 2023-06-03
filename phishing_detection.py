import numpy as np
import pandas as pd
import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from flask import Flask,render_template,request


#Importing dataset
data = pd.read_csv('phishing.csv',delimiter=",")

#Seperating features and labels
X = np.array(data.iloc[: , :-1])
y = np.array(data.iloc[: , -1])


print(type(X))
#Seperating training features, testing features, training labels & testing labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



# classifier = RandomForestClassifier()
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
score = score*100
print(score)


app = Flask(__name__)


@app.route('/',methods=['GET','POST'])


def main():

    if request.method=="POST":
        X_input=request.form.get("url")
        X_new = []
        X_new=feature_extraction.generate_data_set(X_input)
        X_new = np.array(X_new).reshape(1,-1)

        analysis_result = ""

        try:
            prediction = classifier.predict(X_new)
            if prediction == -1:
                analysis_result = "Phishing URL"
            elif prediction == 0:
                analysis_result = "Suspecious"
            else:
                analysis_result = "Good URL"
        except:
            analysis_result = "Phishing URL"
    else:
        analysis_result = ""
        
    return  render_template("index.html",output=analysis_result)

if __name__ == '__main__':
	app.run(debug=True)
