
# Background

This blogpost will serve as a simple end-to-end example of how to train and save your own tensorflow-model
and load it at a later time to do inference in go. 

If you are doing inference in java or any other language
the blogpost will still be useful since the principles are the same for languages with bindings to tensorflow(for instance java).

# Pre reqs

* tensorflow installed
* some familiarity with tensorflows compute-graph 
* Golang is easy to read, so you will probably be able to understand what goes on by just looking at the code. 

# Install go-bindings

To get started with go, you need to install the go-bindings: 

    TF_TYPE="cpu" # Change to "gpu" for GPU support
    TARGET_DIRECTORY='/usr/local'
    curl -L \
       "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.1.0.tar.gz" |
    sudo tar -C $TARGET_DIRECTORY -xz

    sudo ldconfig

    go get github.com/tensorflow/tensorflow/tensorflow/go
    go test github.com/tensorflow/tensorflow/tensorflow/go


(if any of the above fails, find out why on https://www.tensorflow.org/install/install_go )


# Name your tensors and operations

To be able to address specific parts of your tensorflow-graph, you need to give names
to the tensors and operations that you are interested in. In this case, it is the input-tensor and the inference-step I want to address.
 
Here I have named my input-tensor "imageinput" and the inference-step for "infer"

    # the input-tensor
    x = tf.placeholder(tf.float32, [None, 784], name="imageinput")

    # the infer operation
    infer = tf.argmax(y,1, name="infer")

If you are interested in the result of other tensors or operations, be sure to name them as well. For instance, you will probably be interested in 
the accuracy of each inference, so go ahead and give the accuracy-operation a proper name.

# Save your model with a tag

you need to build a builder with tags:

    builder = tf.saved_model.builder.SavedModelBuilder("export2")
    
    builder.add_meta_graph_and_variables(sess,["serve"])
    
    builder.save()

# Load the model in go




## similar blogposts

I have not found any documentation or end-to-end example of what I have described here, but there is a nice post about working with golang and tensorflow in general here:
https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/?utm_source=golangweekly&utm_medium=email
