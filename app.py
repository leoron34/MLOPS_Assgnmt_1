from flask import Flask, jsonify
import model_v2

app = Flask(__name__)

@app.route('/train_decision_tree', methods=['POST'])
def train_and_log_decision_tree():
    evaluation = model_v2.run_decision_tree_experiment()
    return jsonify(evaluation)

@app.route('/train_svm', methods=['POST'])
def train_and_log_svm():
    evaluation = model_v2.run_svm_experiment()
    return jsonify(evaluation)

if __name__ == '__main__':
    app.run(debug=True)
