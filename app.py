from flask import Flask, render_template, request
from facerec.recognition.recognize import recognize_image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognize_image_file", methods=["GET", "POST"])
def recognize_image_file():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save the file to the facerec directory
            file.save("facerec/image.jpg")

            # Call recognize_image function and pass the image path
            recognized_class = recognize_image("facerec/image.jpg")

            # Render the template with the recognized class
            return render_template("index.html", recognized_class=recognized_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
