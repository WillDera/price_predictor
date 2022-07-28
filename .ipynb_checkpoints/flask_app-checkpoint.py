from fileinput import filename
from flask import Flask
from flask_cors import CORS
from requests import request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Normalization
def minmaxscalar(X, min, max):
    omax, omin = X.max(axis=0), X.min(axis=0)

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

    return X_scaled, omax, omin


def inverse_scalar(X, omax, omin, min, max):
    X = X - min
    X = X / (max - min)

    p1 = X + omin
    p2 = omax - omin
    X = X * (omax - omin)
    X += omin

    return X


# Get required Columns


def getColumnsData(df, cols):
    print("Retrieving", " ".join(cols), "Column(s)")
    return df[cols]


def getRequiredColumns(df):
    res = []
    dateColName = None
    closeColName = None

    for col in df.columns:
        if ("date" in col.lower()) or ("time" in col.lower()):
            dateColName = col
            break

    for col in df.columns:
        if "open" in col.lower():
            res.append(col)
            break

    for col in df.columns:
        if "open" in col.lower():
            res.append(col)
            break

    for col in df.columns:
        if "low" in col.lower():
            res.append(col)
            break

    for col in df.columns:
        if "high" in col.lower():
            res.append(col)
            break

    for col in df.columns:
        if ("close" in col.lower()) and (
            "adj" not in col.lower() and ("prev" not in col.lower())
        ):
            res.append(col)
            closeColName = col
            break

    for col in df.columns:
        if ("open" in col.lower()) or ("turnover" in col.lower()):
            res.append(col)
            break


# LSTM MODEL
def LSTM_Cell(
    inputs,
    init_h,
    init_c,
    kernel,
    recurrent_kernel,
    bias,
    mask,
    time_major,
    go_backwards,
    sequence_lengths,
    zero_output_for_mask,
):
    input_length = inputs.shape[1]

    def operations(cell_inputs, cell_states):

        h_tm1 = cell_states[0]
        c_tm1 = cell_states[1]

        z = K.dot(cell_inputs, kernel)


app = Flask("Stock Price Prediction")
CORS(app)

df = None
cols, dateColName, closeColName = None, None, None
train_size = 0.60
totalEpochs = 2

session = {
    "training": {
        "status": "ready",
        "fileUploaded": False,
        "fileName": None,
        "totalEpochs": totalEpochs,
    },
    "prediction": {"status": "ready", "preTrainedModelNames": None},
}


def updateEpochs(epoch):
    global session

    session["training"]["epochs"] = epoch + 1


from api import *


@app.route("/")
def index():
    return "Welcome to Stock Price Prediction API"


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        global session, df, cols, dateColName, closeColName

        df = pd.read_csv(request.files["file"])
        cols, dateColName, closeColName = getRequiredColumns(df)

        # print (df[[dateColName] + cols].head().values)
        dfColVals = []
        dfDateVals = []
        dfCloseVals = []
        for row in df[[dateColName] + cols].values:
            dfColVals.append(list(row))
            dfCloseVals.append(row[4])
            dfDateVals.append(row[0])

        session["training"]["fileUploaded"] = True
        session["training"]["fileName"] = request.files["file"].filename[:-4]
        session["training"]["cols"] = [dateColName] + cols
        session["training"]["dfColVals"] = dfColVals
        session["training"]["dfDateVals"] = dfCloseVals
        session["training"]["dfDateVals"] = dfDateVals

        return session["training"]
    else:
        return "This API accepts only POST requests"


@app.route("/startTraining", methods=["POST", "GET"])
def startTraining():
    if request.method == "POST":
        global session, df

        filename = request.form["fileName"]

        df.to_csv("datasets/" + fileName + ".csv")
        session["training"]["status"] = "training"
        session["training"]["epochs"] = 0

        model = LSTMAlgorithm(
            fileName, train_size, totalEpochs, updateEpochs=updateEpochs
        )

        session["training"]["status"] = "trainingCompleted"

        return session["training"]
    else:
        return "This API accepts only POST requests"


@app.route("/trainingStatus", methods=["POST", "GET"])
def getPreTrainedModels():
    if request.method == "POST":
        global session

        files = glob.glob("./pretrained/*.H5")

        for i in range(len(files)):
            files[i] = files[i][13:-3]

            session["prediction"]["preTrainedModelNames"] = files

            return session["prediction"]
    else:
        return "This API accepts only POST requests"


@app.route("/getPredictions", methods=["POST", "GET"])
def getPredictions():
    if request.method == "POST":
        global session

        modelName = request.form["modelName"]
        session["prediction"]["modelName"] = modelName
        modelData = getPredictionsFromModel(modelName, train_size)
        session["prediction"]["modelData"] = modelData

        return session["prediction"]
    else:
        return "This API accepts only POST requests"


if __name__ == "__main__":
    debug = False
    port = 8081

    app.run(debug=debug, port=port)
