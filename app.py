import logging
import os
from http import HTTPStatus

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from main_function import AutoClassification

app = Flask(__name__)
logging.info('working0')

# # Configure the upload folder
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        logging.info('working3')
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Get the image file from the request
        image = request.files['image']
        # image = request.files.get('image')

        # Check if the image file is empty
        if image.filename == '':
            return jsonify({'error': 'No image file provided'}), 400
        #
        # # Save the image to the upload folder
        # filename = secure_filename(image.filename)
        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # image.save(image_path)

        # Perform image classification
        predicted_gender, predicted_sleeve = AutoClassification().prediction(image)

        # Return the predicted class
        return jsonify({'GENDER': predicted_gender, 'CLASS': predicted_sleeve}), 200

    except Exception as e:

        message = "Error While Performing Classification: {}".format(e)
        print(message)
        response_data = {
            "message": message,
            "status_code": HTTPStatus.REQUEST_TIMEOUT
        }
        return jsonify(response_data), HTTPStatus.REQUEST_TIMEOUT


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
