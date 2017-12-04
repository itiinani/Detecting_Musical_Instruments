

from flask import render_template
from flask import Flask, request, redirect
from Demo import fileDetection

app = Flask(__name__)
@app.route("/")
def hello_world():

    return render_template("/pages/charts/chartjs.html")

@app.route("/results")
def results():
    return render_template("/pages/charts/results.html")

@app.route("/cons_results")
def cons_results():
    return render_template("/pages/charts/consolidated_result.html")

@app.route("/uploadmusicfile", methods=['GET','POST'])
def uploadfile_and_detect():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'musicfile' not in request.files:
           # flash('No file part')
            return redirect(request.url)
        file = request.files['musicfile']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        fileDetection(file.filename)
    return render_template("/pages/charts/chartjs.html")



if __name__ == "__main__":
   app.run("127.0.0.1",3013)