from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# RECOMENDACIONES
df = pd.DataFrame({'Cliente': [1, 1, 2, 2, 3, 3], 'Producto': ['A', 'B', 'A', 'C', 'B', 'C']})
df_pivot = df.pivot_table(index='Cliente', columns='Producto', aggfunc=lambda x: 1, fill_value=0)
frequent_items = apriori(df_pivot, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
recomendaciones = rules[['antecedents', 'consequents', 'support', 'confidence']].to_html()

# MODELO DE PREDICCIÓN
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 0, 1])
model = LogisticRegression().fit(X_train, y_train)

# CHATBOT
chatbot = ChatBot("Soporte")
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.spanish")

@app.route("/", methods=["GET", "POST"])
def home():
    prediccion = None
    if request.method == "POST":
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])
        prediccion = model.predict(np.array([[x1, x2]]))[0]
    return render_template("index.html", recomendaciones=recomendaciones, prediccion=prediccion)

@app.route("/chat", methods=["POST"])
def chat():
    mensaje = request.json["mensaje"]
    respuesta = chatbot.get_response(mensaje)
    return jsonify({"respuesta": str(respuesta)})

if __name__ == "__main__":
    app.run(debug=True)
 
