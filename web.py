

from flask import render_template
from flask import Flask, request, redirect
from Demo import fileDetection

app = Flask(__name__)
@app.route("/")
def hello_world():

    return render_template("/pages/charts/chartjs.html")

@app.route("/PC")
def parallel_coordinates_graph():
    return render_template("/pages/charts/parallel_coordinates.html")

@app.route("/FD")
def forceDirected_graph():
    return render_template("/pages/charts/forceDirected.html")

@app.route("/AT")
def allTranscripts_graph():
    return render_template("/pages/charts/allCharts.html")

@app.route("/uploadmusicfile", methods=['GET','POST'])
def uploadfile_and_detect():
     fileDetection()
     return render_template("/pages/charts/chartjs.html")



if __name__ == "__main__":
   app.run("127.0.0.1",3012)