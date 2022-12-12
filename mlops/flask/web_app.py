from flask import Flask, render_template, request

from transformers import RobertaTokenizer, T5ForConditionalGeneration

def predict_code_description(text):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=20)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
 
app = Flask(__name__)
 
@app.route("/code", methods=["GET", "POST"])
def index():
    desc = ""
    if (request.method == "POST"):
        print(request.json)
        code_inp = request.json.get('code')
        print(type(code_inp))
        desc = predict_code_description(code_inp)
        print(desc)
        # desc = "Description of code is: Convert string to int"
        return {'response': desc}
    return render_template("index.html", desc=desc)
 
if __name__ == "__main__":
    app.run(debug=True)
