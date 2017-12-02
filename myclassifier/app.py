import os
import tensorflow as tf
import sys
import time
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename


# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = '/home/garima/Documents/myclassifier/static/uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

label_lines = [line.rstrip() for line 
               in tf.gfile.GFile("/home/garima/Documents/myclassifier/retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("/home/garima/Documents/myclassifier/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('indexKanishka.html')
    return redirect(url_for('static', filename='background.png'))


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file',
                                filename=filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/home/garima/Documents/myclassifier/static/uploads/<filename>')
def uploaded_file(filename):
    start_time = 0
    start_time = time.time()
    # change this as you see fit
    #image_path = sys.argv[1]
    image_path="/home/garima/Documents/myclassifier/static/uploads/"+filename
    strng = " "
    strng = strng + str(time.time() - start_time)+ "|"
    resultlist=[" "]
    #resultlist.append( str(time.time() - start_time) )
    # label_lines = [line.rstrip() for line 
    #                in tf.gfile.GFile("/home/nishith/tensorflow_image_classifier/Trained Model/retrained_labels.txt")]

    # # Unpersists graph from file
    # with tf.gfile.FastGFile("/home/nishith/tensorflow_image_classifier/Trained Model/retrained_graph.pb", 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     _ = tf.import_graph_def(graph_def, name='')

    strng = strng + str(time.time() - start_time)+ "|"
    #resultlist.append( str(time.time() - start_time) )
    # with tf.Session() as sess:
    #     # Feed the image_data as input to the graph and get first prediction
    #     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
          
    start_time = time.time()
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    print (image_path)
    predictions = sess.run(softmax_tensor,
            {'DecodeJpeg/contents:0': image_data})
    
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    firstElt = top_k[0];
        
    str2=" "

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if (score>0.70 and human_string == 'aadhar'):
            str2 = str2 + " You have successfully registered."
        elif (score>0.70 and human_string == 'unacceptable_id'):
            str2 =  "Sorry, we couldn't authenticate your ID."
        #return human_string
        #print (node_id)
        #print('%s (score = %.5f)' % (human_string, score))
    if (str2 ==  " "):
        strng = "Sorry, we couldn't authenticate your ID."
        resultlist = ["Sorry, we couldn't authenticate your ID."]
        #return render_template('upload_file.html', strng=strng)
        return render_template('upload_file.html', resultlist=resultlist)
        
    #strng= strng +" | "+ human_string +"("+ str(score*100)+"%)"
    strng = strng + str2 + "| Time = "+ str(time.time() - start_time) + "sec"
    resultlist.append(str2)
    #resultlist.append("Time = "+ str(time.time() - start_time))
    #return strng
    #return render_template('upload_file.html', strng=strng)
    return render_template('upload_file.html', resultlist=resultlist)
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
    app.run(
        #host="0.0.0.0",
        #port=int("80"),
        debug=True
)
